// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).
//
// BSkipList is a B-skiplist: each node holds several sorted keys, and a key
// is promoted to the next level with probability 1/kBranch, same idea as
// Yicong Luo et al., "Bridging Cache-Friendliness and Concurrency: A
// Locality-Optimized In-Memory B-Skiplist" (ICPP 2025).
//
// This is not the paper's reader-writer lock protocol. RocksDB memtable
// reads run alongside a single writer (Insert is externally synchronized)
// and must not take a lock or bump a reader count. Writers publish node
// updates with a seqlock; readers retry when the version changes. Nodes are
// never freed until the list is destroyed, so a reader that loads a node
// pointer can always dereference it.
//
// Ordering is bytewise on the user key, then descending internal-key footer
// (sequence and type). That matches InternalKeyComparator with
// BytewiseComparator. The caller must refuse any other comparator.
//
// The first 16 bytes of the user key and the footer live in the node. Keys
// of that size, which is db_bench's default, are compared without touching
// the memtable entry.

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <unordered_set>
#include <vector>

#include "memory/allocator.h"
#include "port/likely.h"
#include "port/port.h"
#include "rocksdb/slice.h"
#include "util/coding.h"
#include "util/math.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

class BSkipList {
  struct Node;

 public:
  static constexpr uint32_t kFanout = 16;
  // Bytes compared as integers before a search reads the memtable entry.
  // The first 16 are the common case. The next 32 cover keys that share a
  // long prefix; those used to miss into the arena and memcmp the whole key.
  static constexpr uint32_t kInlineUserKey = 16;
  static constexpr uint32_t kExtraWords = 4;
  static constexpr uint32_t kPrefixBytes = kInlineUserKey + kExtraWords * 8;
  static constexpr int kMaxHeight = 8;
  static constexpr uint32_t kBranch = 16;

  explicit BSkipList(Allocator* allocator);
  BSkipList(const BSkipList&) = delete;
  BSkipList& operator=(const BSkipList&) = delete;

  char* AllocateKey(size_t key_size);

  // REQUIRES: no concurrent Insert or InsertConcurrently. Readers may run.
  // Returns false when an equal internal key is already present.
  bool Insert(const char* key);

  // Like Insert, but safe against other InsertConcurrently calls and readers.
  // Writers that touch different nodes run together. A node is claimed with a
  // compare-and-swap on its seqlock, then the key is rechecked. Insert and
  // InsertConcurrently must not run at the same time.
  bool InsertConcurrently(const char* key);

  bool Contains(const char* key) const;

  // `start_ikey` and `end_ikey` are internal keys (user key + footer), not
  // length-prefixed memtable entries. The result is an estimate of how many
  // entries are in [start, end).
  uint64_t ApproximateNumEntries(const Slice& start_ikey,
                                 const Slice& end_ikey) const;

  void UniqueRandomSample(uint64_t target_sample_size,
                          std::unordered_set<const char*>* entries) const;

  void TEST_Validate() const;

  class Iterator {
   public:
    explicit Iterator(const BSkipList* list);

    bool Valid() const { return key_ != nullptr; }
    const char* key() const {
      assert(Valid());
      return key_;
    }
    void Next();
    void Prev();
    void Seek(const char* target);
    void SeekForPrev(const char* target);
    void SeekToFirst();
    void SeekToLast();

   private:
    void Remember(BSkipList::Node* leaf, uint32_t seq, uint32_t count,
                  BSkipList::Node* next);

    const BSkipList* list_;
    BSkipList::Node* leaf_;
    uint32_t index_;
    const char* key_;
    // Seqlock snapshot of `leaf_`, so Next can walk the rest of the node
    // without repeating the fence. Invalid when cached_leaf_ != leaf_.
    BSkipList::Node* cached_leaf_;
    BSkipList::Node* cached_next_;
    uint32_t cached_seq_;
    uint32_t cached_count_;
  };

 private:
  static constexpr uint32_t kScaledInverseBranch =
      (static_cast<uint32_t>(Random::kMaxNext) + 1u) / kBranch;

  struct KeyView {
    const char* entry;
    const char* user;
    uint32_t user_len;
    uint64_t footer;
    // User-key bytes in memcmp order, zero-padded, byte-swapped on little
    // endian so an integer compare matches memcmp. `word0`/`word1` are the
    // first 16 bytes. `extra` is bytes [16, 48).
    uint64_t word0;
    uint64_t word1;
    uint64_t extra[kExtraWords];
  };

  struct SlotView {
    const char* entry;
    uint32_t user_len;
    uint64_t footer;
    uint64_t word0;
    uint64_t word1;
    uint64_t extra[kExtraWords];
    char user[kInlineUserKey];
  };

  struct Slot {
    // word0 is first: FindRank loads it for every occupied slot.
    std::atomic<uint64_t> word0;
    std::atomic<uint64_t> word1;
    std::atomic<uint64_t> footer;
    std::atomic<uint32_t> user_len;
    std::atomic<const char*> entry;

    Slot() : word0(0), word1(0), footer(0), user_len(0), entry(nullptr) {}
  };

  struct Node {
    std::atomic<uint32_t> seq;
    std::atomic<uint32_t> count;
    const uint32_t level;
    std::atomic<Node*> next;
    std::atomic<Node*> prev;
    std::atomic<uint32_t> next_user_len;
    std::atomic<uint64_t> next_footer;
    std::atomic<uint64_t> next_word0;
    std::atomic<uint64_t> next_word1;
    // Slots stay here, at the same offset as a node without the long-key
    // prefix, so the word0 scan stays cache-line aligned.
    Slot slots[kFanout];
    std::atomic<uint64_t> next_extra[kExtraWords];
    // Bytes [16, 48) of each slot, `kFanout * kExtraWords` words. Null when
    // every key in the node is at most 16 bytes, so short-key nodes stay the
    // same size and the word0 scan keeps its stride.
    std::atomic<uint64_t>* extra;
    // Immutable after the node is published. Null on leaves. Elements are
    // mutated only inside the seqlock.
    std::atomic<Node*>* children;

    explicit Node(uint32_t lvl)
        : seq(0),
          count(0),
          level(lvl),
          next(nullptr),
          prev(nullptr),
          next_user_len(0),
          next_footer(0),
          next_word0(0),
          next_word1(0),
          extra(nullptr),
          children(nullptr) {
      for (uint32_t i = 0; i < kExtraWords; ++i) {
        next_extra[i].store(0, std::memory_order_relaxed);
      }
    }
  };

  static_assert(sizeof(Node) % alignof(std::atomic<Node*>) == 0,
                "children array must be aligned");

  struct Step {
    Node* node;
    uint32_t rank;
  };

  Allocator* const allocator_;
  Node* headers_[kMaxHeight];
  std::atomic<Node*> tail_;
  // Highest built level, in [1, kMaxHeight]. Readers start here so empty
  // upper levels are not walked. Single-writer, published with release.
  std::atomic<int> height_;
  std::atomic<uint64_t> num_entries_;

  static KeyView ParseEntry(const char* entry);
  static KeyView ParseInternalKey(const Slice& ikey);
  static void PackUserWords(const char* user, uint32_t user_len, uint64_t* word0,
                            uint64_t* word1);
  static uint64_t PackWordAt(const char* user, uint32_t user_len, uint32_t off);
  static void PackExtra(const char* user, uint32_t user_len, uint64_t* extra);
  static void StoreUserWords(std::atomic<uint64_t>* word0,
                             std::atomic<uint64_t>* word1, const char* user,
                             uint32_t user_len);
  static void LoadUserWords(uint64_t word0, uint64_t word1, char* dst);
  static int CmpPacked(uint64_t a, uint64_t b);
  static int CmpFooter(uint64_t a, uint64_t b);

  Node* NewNode(uint32_t level);
  void WriteSentinel(Node* node);
  void EnsureExtra(Node* node) const;
  void WriteSlot(Node* node, uint32_t index, const KeyView& key) const;
  void WriteSlot(Node* node, uint32_t index, const SlotView& src) const;
  void CopySlot(Node* dst, uint32_t dst_index, const Node* src,
                uint32_t src_index) const;
  SlotView ReadSlot(const Node* node, uint32_t index) const;
  int CompareFull(const char* a, uint32_t a_len, uint64_t a_foot, const char* b,
                  uint32_t b_len, uint64_t b_foot) const;
  // REQUIRES: the first kPrefixBytes user-key bytes already compared equal.
  int CompareSuffix(const char* a, uint32_t a_len, uint64_t a_foot, const char* b,
                    uint32_t b_len, uint64_t b_foot) const;
  // REQUIRES: word0 and word1 already compared equal.
  int CmpPastInline(const KeyView& key, const uint64_t* extra, uint32_t other_len,
                    uint64_t other_foot, const char* other_user) const;
  const char* FullUser(const SlotView& slot) const;
  int CompareKeyToSlot(const KeyView& key, const SlotView& slot) const;
  int CompareKeyToNextHeader(const KeyView& key, const Node* node) const;
  void StoreNextHeader(Node* node, const Node* next_node);
  uint32_t FindRank(const Node* node, uint32_t count, const KeyView& key,
                    SlotView* at) const;
  Node* WalkRight(Node* node, const KeyView& key) const;
  int RandomHeight() const;
  void BeginWrite(Node* node) const;
  void EndWrite(Node* node) const;
  bool TryLockNode(Node* node) const;
  void LockNode(Node* node) const;
  bool KeyBelongs(Node* node, const KeyView& key) const;
  Node* Locate(const KeyView& key, int target_level) const;
  void SpliceLocked(Node* left, Node* neu, uint32_t new_left_count);
  void FixRightPrev(Node* neu, Node* right);
  void PublishTail(Node* candidate);
  Node* PromotionSplitLocked(Node* node, uint32_t rank, const KeyView& key,
                             Node* new_child);
  void SplitAndInsertLocked(Node* node, uint32_t rank, const KeyView& key,
                            Node* new_child);
  void ShiftInsertLocked(Node* node, uint32_t pos, const KeyView& key,
                         Node* child);
  bool TryAppendConcurrent(const KeyView& key);
  // Null when `key` is already present at this level.
  Node* PromoteConcurrent(const KeyView& key, int level, Node* new_child);
  bool InsertAtLevelConcurrent(const KeyView& key, int level, Node* child);
  void PublishHeight(int height);
  bool TryAppend(const KeyView& key);
  void SpliceAfter(Node* left, Node* neu, uint32_t new_left_count);
  Node* PromotionSplit(Node* node, uint32_t rank, const KeyView& key,
                       Node* new_child);
  void SplitAndInsert(Node* node, uint32_t rank, const KeyView& key,
                      Node* new_child);
  void ShiftInsert(Node* node, uint32_t pos, const KeyView& key, Node* child);
  bool Descend(const KeyView& key, Node** leaf, uint32_t* index,
               bool* equal) const;
  bool ReadEntryAt(Node* node, uint32_t index, const char** entry,
                   Node** next) const;

  friend class Iterator;
};

inline void BSkipList::PackUserWords(const char* user, uint32_t user_len,
                                     uint64_t* word0, uint64_t* word1) {
  uint64_t first = 0;
  uint64_t second = 0;
  const uint32_t n = user_len < kInlineUserKey ? user_len : kInlineUserKey;
  if (n > 0) {
    memcpy(&first, user, n < 8 ? n : 8);
  }
  if (n > 8) {
    memcpy(&second, user + 8, n - 8);
  }
  if (port::kLittleEndian) {
    first = EndianSwapValue(first);
    second = EndianSwapValue(second);
  }
  *word0 = first;
  *word1 = second;
}

inline uint64_t BSkipList::PackWordAt(const char* user, uint32_t user_len,
                                      uint32_t off) {
  uint64_t word = 0;
  if (user_len > off) {
    const uint32_t n = user_len - off < 8 ? user_len - off : 8;
    memcpy(&word, user + off, n);
    if (port::kLittleEndian) {
      word = EndianSwapValue(word);
    }
  }
  return word;
}

inline void BSkipList::PackExtra(const char* user, uint32_t user_len,
                                 uint64_t* extra) {
  if (user_len <= kInlineUserKey) {
    for (uint32_t t = 0; t < kExtraWords; ++t) {
      extra[t] = 0;
    }
    return;
  }
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    extra[t] = PackWordAt(user, user_len, kInlineUserKey + t * 8);
  }
}

inline BSkipList::KeyView BSkipList::ParseEntry(const char* entry) {
  Slice ikey = GetLengthPrefixedSlice(entry);
  assert(ikey.size() >= 8);
  KeyView view;
  view.entry = entry;
  view.user = ikey.data();
  view.user_len = static_cast<uint32_t>(ikey.size() - 8);
  view.footer = DecodeFixed64(ikey.data() + view.user_len);
  PackUserWords(view.user, view.user_len, &view.word0, &view.word1);
  PackExtra(view.user, view.user_len, view.extra);
  return view;
}

inline BSkipList::KeyView BSkipList::ParseInternalKey(const Slice& ikey) {
  assert(ikey.size() >= 8);
  KeyView view;
  view.entry = nullptr;
  view.user = ikey.data();
  view.user_len = static_cast<uint32_t>(ikey.size() - 8);
  view.footer = DecodeFixed64(ikey.data() + view.user_len);
  PackUserWords(view.user, view.user_len, &view.word0, &view.word1);
  PackExtra(view.user, view.user_len, view.extra);
  return view;
}

inline void BSkipList::StoreUserWords(std::atomic<uint64_t>* word0,
                                      std::atomic<uint64_t>* word1,
                                      const char* user, uint32_t user_len) {
  uint64_t first = 0;
  uint64_t second = 0;
  PackUserWords(user, user_len, &first, &second);
  word0->store(first, std::memory_order_relaxed);
  word1->store(second, std::memory_order_relaxed);
}

inline void BSkipList::LoadUserWords(uint64_t word0, uint64_t word1,
                                     char* dst) {
  if (port::kLittleEndian) {
    word0 = EndianSwapValue(word0);
    word1 = EndianSwapValue(word1);
  }
  memcpy(dst, &word0, 8);
  memcpy(dst + 8, &word1, 8);
}

inline int BSkipList::CmpPacked(uint64_t a, uint64_t b) {
  if (a < b) {
    return -1;
  }
  if (a > b) {
    return 1;
  }
  return 0;
}

inline int BSkipList::CmpFooter(uint64_t a, uint64_t b) {
  // A larger footer (newer sequence) sorts first.
  if (a > b) {
    return -1;
  }
  if (a < b) {
    return 1;
  }
  return 0;
}

inline BSkipList::BSkipList(Allocator* allocator)
    : allocator_(allocator),
      headers_(),
      tail_(nullptr),
      height_(1),
      num_entries_(0) {
  for (int level = 0; level < kMaxHeight; ++level) {
    headers_[level] = NewNode(static_cast<uint32_t>(level));
    WriteSentinel(headers_[level]);
  }
  for (int level = 1; level < kMaxHeight; ++level) {
    headers_[level]->children[0].store(headers_[level - 1],
                                       std::memory_order_relaxed);
  }
  tail_.store(headers_[0], std::memory_order_relaxed);
}

inline char* BSkipList::AllocateKey(size_t key_size) {
  return allocator_->Allocate(key_size);
}

inline BSkipList::Node* BSkipList::NewNode(uint32_t level) {
  size_t bytes = sizeof(Node);
  if (level > 0) {
    bytes += sizeof(std::atomic<Node*>) * kFanout;
  }
  char* mem = allocator_->AllocateAligned(bytes);
  Node* node = new (mem) Node(level);
  if (level > 0) {
    // Construct each pointer in place. Array placement-new is not portable:
    // some standard libraries store a size cookie in front of the array.
    void* raw = mem + sizeof(Node);
    node->children = static_cast<std::atomic<Node*>*>(raw);
    for (uint32_t i = 0; i < kFanout; ++i) {
      new (node->children + i) std::atomic<Node*>(nullptr);
    }
  }
  return node;
}

inline void BSkipList::WriteSentinel(Node* node) {
  node->slots[0].entry.store(nullptr, std::memory_order_relaxed);
  node->slots[0].footer.store(0, std::memory_order_relaxed);
  node->slots[0].user_len.store(0, std::memory_order_relaxed);
  node->slots[0].word0.store(0, std::memory_order_relaxed);
  node->slots[0].word1.store(0, std::memory_order_relaxed);
  node->count.store(1, std::memory_order_relaxed);
}

inline void BSkipList::EnsureExtra(Node* node) const {
  if (node->extra != nullptr) {
    return;
  }
  const size_t n = static_cast<size_t>(kFanout) * kExtraWords;
  char* raw = allocator_->AllocateAligned(n * sizeof(std::atomic<uint64_t>));
  std::atomic<uint64_t>* words =
      static_cast<std::atomic<uint64_t>*>(static_cast<void*>(raw));
  for (size_t i = 0; i < n; ++i) {
    new (words + i) std::atomic<uint64_t>(0);
  }
  node->extra = words;
}

inline void BSkipList::WriteSlot(Node* node, uint32_t index,
                                 const KeyView& key) const {
  Slot* dst = &node->slots[index];
  dst->entry.store(key.entry, std::memory_order_relaxed);
  dst->footer.store(key.footer, std::memory_order_relaxed);
  dst->user_len.store(key.user_len, std::memory_order_relaxed);
  StoreUserWords(&dst->word0, &dst->word1, key.user, key.user_len);
  if (key.user_len <= kInlineUserKey && node->extra == nullptr) {
    return;
  }
  EnsureExtra(node);
  std::atomic<uint64_t>* row = node->extra + index * kExtraWords;
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    row[t].store(key.extra[t], std::memory_order_relaxed);
  }
}

inline void BSkipList::WriteSlot(Node* node, uint32_t index,
                                 const SlotView& src) const {
  Slot* dst = &node->slots[index];
  dst->word0.store(src.word0, std::memory_order_relaxed);
  dst->word1.store(src.word1, std::memory_order_relaxed);
  dst->footer.store(src.footer, std::memory_order_relaxed);
  dst->user_len.store(src.user_len, std::memory_order_relaxed);
  dst->entry.store(src.entry, std::memory_order_relaxed);
  if (src.user_len <= kInlineUserKey && node->extra == nullptr) {
    return;
  }
  EnsureExtra(node);
  std::atomic<uint64_t>* row = node->extra + index * kExtraWords;
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    row[t].store(src.extra[t], std::memory_order_relaxed);
  }
}

inline void BSkipList::CopySlot(Node* dst, uint32_t dst_index, const Node* src,
                                uint32_t src_index) const {
  const Slot& from = src->slots[src_index];
  Slot* to = &dst->slots[dst_index];
  to->entry.store(from.entry.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  to->footer.store(from.footer.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
  to->user_len.store(from.user_len.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
  to->word0.store(from.word0.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  to->word1.store(from.word1.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  if (src->extra == nullptr) {
    return;
  }
  EnsureExtra(dst);
  const std::atomic<uint64_t>* from_row = src->extra + src_index * kExtraWords;
  std::atomic<uint64_t>* to_row = dst->extra + dst_index * kExtraWords;
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    to_row[t].store(from_row[t].load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
  }
}

inline BSkipList::SlotView BSkipList::ReadSlot(const Node* node,
                                               uint32_t index) const {
  const Slot& src = node->slots[index];
  SlotView view;
  view.word0 = src.word0.load(std::memory_order_relaxed);
  view.word1 = src.word1.load(std::memory_order_relaxed);
  view.footer = src.footer.load(std::memory_order_relaxed);
  view.user_len = src.user_len.load(std::memory_order_relaxed);
  view.entry = src.entry.load(std::memory_order_relaxed);
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    view.extra[t] = 0;
  }
  if (node->extra != nullptr) {
    const std::atomic<uint64_t>* row = node->extra + index * kExtraWords;
    for (uint32_t t = 0; t < kExtraWords; ++t) {
      view.extra[t] = row[t].load(std::memory_order_relaxed);
    }
  }
  LoadUserWords(view.word0, view.word1, view.user);
  return view;
}

inline int BSkipList::CompareFull(const char* a, uint32_t a_len,
                                  uint64_t a_foot, const char* b,
                                  uint32_t b_len, uint64_t b_foot) const {
  const uint32_t n = a_len < b_len ? a_len : b_len;
  int cmp = 0;
  if (n != 0) {
    cmp = memcmp(a, b, n);
  }
  if (cmp != 0) {
    return cmp;
  }
  if (a_len < b_len) {
    return -1;
  }
  if (a_len > b_len) {
    return 1;
  }
  if (a_foot > b_foot) {
    return -1;
  }
  if (a_foot < b_foot) {
    return 1;
  }
  return 0;
}

inline int BSkipList::CompareSuffix(const char* a, uint32_t a_len,
                                    uint64_t a_foot, const char* b,
                                    uint32_t b_len, uint64_t b_foot) const {
  const uint32_t a_from = a_len > kPrefixBytes ? kPrefixBytes : a_len;
  const uint32_t b_from = b_len > kPrefixBytes ? kPrefixBytes : b_len;
  const uint32_t a_left = a_len - a_from;
  const uint32_t b_left = b_len - b_from;
  const uint32_t n = a_left < b_left ? a_left : b_left;
  int cmp = 0;
  if (n != 0) {
    cmp = memcmp(a + a_from, b + b_from, n);
  }
  if (cmp != 0) {
    return cmp;
  }
  if (a_len < b_len) {
    return -1;
  }
  if (a_len > b_len) {
    return 1;
  }
  return CmpFooter(a_foot, b_foot);
}

inline int BSkipList::CmpPastInline(const KeyView& key, const uint64_t* extra,
                                    uint32_t other_len, uint64_t other_foot,
                                    const char* other_user) const {
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    const int c = CmpPacked(key.extra[t], extra[t]);
    if (c != 0) {
      return c;
    }
  }
  if (LIKELY(key.user_len <= kPrefixBytes && other_len <= kPrefixBytes)) {
    if (key.user_len != other_len) {
      return key.user_len < other_len ? -1 : 1;
    }
    return CmpFooter(key.footer, other_foot);
  }
  return CompareSuffix(key.user, key.user_len, key.footer, other_user, other_len,
                       other_foot);
}

inline const char* BSkipList::FullUser(const SlotView& slot) const {
  if (slot.entry == nullptr || slot.user_len <= kInlineUserKey) {
    return slot.user;
  }
  Slice ikey = GetLengthPrefixedSlice(slot.entry);
  return ikey.data();
}

inline int BSkipList::CompareKeyToSlot(const KeyView& key,
                                       const SlotView& slot) const {
  if (slot.entry == nullptr) {
    return 1;
  }
  const int c0 = CmpPacked(key.word0, slot.word0);
  if (c0 != 0) {
    return c0;
  }
  const int c1 = CmpPacked(key.word1, slot.word1);
  if (c1 != 0) {
    return c1;
  }
  if (LIKELY(key.user_len <= kInlineUserKey &&
             slot.user_len <= kInlineUserKey)) {
    if (key.user_len != slot.user_len) {
      return key.user_len < slot.user_len ? -1 : 1;
    }
    return CmpFooter(key.footer, slot.footer);
  }
  const char* other = nullptr;
  if (key.user_len > kPrefixBytes || slot.user_len > kPrefixBytes) {
    other = FullUser(slot);
  }
  return CmpPastInline(key, slot.extra, slot.user_len, slot.footer, other);
}

inline int BSkipList::CompareKeyToNextHeader(const KeyView& key,
                                             const Node* node) const {
  const uint64_t nw0 = node->next_word0.load(std::memory_order_relaxed);
  const int c0 = CmpPacked(key.word0, nw0);
  if (c0 != 0) {
    return c0;
  }
  const uint64_t nw1 = node->next_word1.load(std::memory_order_relaxed);
  const int c1 = CmpPacked(key.word1, nw1);
  if (c1 != 0) {
    return c1;
  }
  const uint32_t nlen = node->next_user_len.load(std::memory_order_relaxed);
  const uint64_t nfoot = node->next_footer.load(std::memory_order_relaxed);
  if (LIKELY(key.user_len <= kInlineUserKey && nlen <= kInlineUserKey)) {
    if (key.user_len != nlen) {
      return key.user_len < nlen ? -1 : 1;
    }
    return CmpFooter(key.footer, nfoot);
  }
  uint64_t extra[kExtraWords];
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    extra[t] = node->next_extra[t].load(std::memory_order_relaxed);
  }
  const char* other = nullptr;
  if (key.user_len > kPrefixBytes || nlen > kPrefixBytes) {
    // The entry is needed only when bytes [0, 48) match.
    bool prefix_equal = true;
    for (uint32_t t = 0; t < kExtraWords; ++t) {
      if (key.extra[t] != extra[t]) {
        prefix_equal = false;
        break;
      }
    }
    if (prefix_equal) {
      Node* next = node->next.load(std::memory_order_relaxed);
      assert(next != nullptr);
      const char* entry = next->slots[0].entry.load(std::memory_order_relaxed);
      assert(entry != nullptr);
      other = GetLengthPrefixedSlice(entry).data();
    }
  }
  return CmpPastInline(key, extra, nlen, nfoot, other);
}

inline void BSkipList::StoreNextHeader(Node* node, const Node* next_node) {
  const Slot& slot0 = next_node->slots[0];
  node->next_user_len.store(slot0.user_len.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
  node->next_footer.store(slot0.footer.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
  node->next_word0.store(slot0.word0.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
  node->next_word1.store(slot0.word1.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    uint64_t word = 0;
    if (next_node->extra != nullptr) {
      word = next_node->extra[t].load(std::memory_order_relaxed);
    }
    node->next_extra[t].store(word, std::memory_order_relaxed);
  }
}

inline uint32_t BSkipList::FindRank(const Node* node, uint32_t count,
                                    const KeyView& key, SlotView* at) const {
  assert(count > 0);
  assert(count <= kFanout);
  const uint64_t kw0 = key.word0;
  const uint64_t kw1 = key.word1;
  uint32_t rank = 0;
  for (uint32_t i = 1; i < count; ++i) {
    const uint64_t sw0 = node->slots[i].word0.load(std::memory_order_relaxed);
    if (UNLIKELY(kw0 == sw0)) {
      const uint64_t sw1 = node->slots[i].word1.load(std::memory_order_relaxed);
      if (kw1 < sw1) {
        break;
      }
      if (kw1 > sw1) {
        rank = i;
        continue;
      }
      const uint32_t slen =
          node->slots[i].user_len.load(std::memory_order_relaxed);
      const uint64_t sfoot =
          node->slots[i].footer.load(std::memory_order_relaxed);
      int cmp;
      if (LIKELY(key.user_len <= kInlineUserKey && slen <= kInlineUserKey)) {
        if (key.user_len != slen) {
          cmp = key.user_len < slen ? -1 : 1;
        } else {
          cmp = CmpFooter(key.footer, sfoot);
        }
      } else {
        uint64_t extra[kExtraWords];
        for (uint32_t t = 0; t < kExtraWords; ++t) {
          extra[t] = 0;
        }
        if (node->extra != nullptr) {
          const std::atomic<uint64_t>* row = node->extra + i * kExtraWords;
          for (uint32_t t = 0; t < kExtraWords; ++t) {
            extra[t] = row[t].load(std::memory_order_relaxed);
          }
        }
        const char* other = nullptr;
        bool need_suffix = key.user_len > kPrefixBytes || slen > kPrefixBytes;
        if (need_suffix) {
          bool prefix_equal = true;
          for (uint32_t t = 0; t < kExtraWords; ++t) {
            if (key.extra[t] != extra[t]) {
              prefix_equal = false;
              break;
            }
          }
          if (prefix_equal) {
            const char* entry =
                node->slots[i].entry.load(std::memory_order_relaxed);
            other = GetLengthPrefixedSlice(entry).data();
          }
        }
        cmp = CmpPastInline(key, extra, slen, sfoot, other);
      }
      if (cmp < 0) {
        break;
      }
      rank = i;
      continue;
    }
    if (kw0 < sw0) {
      break;
    }
    rank = i;
  }
  *at = ReadSlot(node, rank);
  return rank;
}

inline BSkipList::Node* BSkipList::WalkRight(Node* node,
                                             const KeyView& key) const {
  for (;;) {
    Node* next = node->next.load(std::memory_order_relaxed);
    if (next == nullptr) {
      return node;
    }
    if (CompareKeyToNextHeader(key, node) < 0) {
      return node;
    }
    node = next;
  }
}

inline int BSkipList::RandomHeight() const {
  Random* rnd = Random::GetTLSInstance();
  int height = 0;
  while (height + 1 < kMaxHeight && rnd->Next() < kScaledInverseBranch) {
    ++height;
  }
  return height;
}

inline void BSkipList::BeginWrite(Node* node) const {
  const uint32_t seq = node->seq.load(std::memory_order_relaxed);
  assert((seq & 1u) == 0);
  node->seq.store(seq + 1, std::memory_order_release);
}

inline void BSkipList::EndWrite(Node* node) const {
  const uint32_t seq = node->seq.load(std::memory_order_relaxed);
  assert((seq & 1u) != 0);
  node->seq.store(seq + 1, std::memory_order_release);
}

inline bool BSkipList::TryLockNode(Node* node) const {
  uint32_t seq = node->seq.load(std::memory_order_acquire);
  if ((seq & 1u) != 0) {
    return false;
  }
  return node->seq.compare_exchange_strong(
      seq, seq + 1, std::memory_order_acq_rel, std::memory_order_acquire);
}

inline void BSkipList::LockNode(Node* node) const {
  while (UNLIKELY(!TryLockNode(node))) {
    port::AsmVolatilePause();
  }
}

inline bool BSkipList::KeyBelongs(Node* node, const KeyView& key) const {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  if (count == 0 || count > kFanout) {
    return false;
  }
  const SlotView head = ReadSlot(node, 0);
  if (head.entry != nullptr && CompareKeyToSlot(key, head) < 0) {
    return false;
  }
  if (node->next.load(std::memory_order_relaxed) != nullptr &&
      CompareKeyToNextHeader(key, node) >= 0) {
    return false;
  }
  return true;
}

inline BSkipList::Node* BSkipList::Locate(const KeyView& key,
                                          int target_level) const {
  const int top = height_.load(std::memory_order_acquire);
  if (target_level >= top) {
    return headers_[target_level];
  }
  Node* node = headers_[top - 1];
  for (int level = top - 1; level >= target_level; --level) {
    node = WalkRight(node, key);
    if (node->level != static_cast<uint32_t>(level)) {
      return nullptr;
    }
    if (level == target_level) {
      return node;
    }
    const uint32_t count = node->count.load(std::memory_order_acquire);
    if (count == 0 || count > kFanout || node->children == nullptr) {
      return nullptr;
    }
    SlotView at;
    const uint32_t rank = FindRank(node, count, key, &at);
    Node* child = node->children[rank].load(std::memory_order_acquire);
    if (child == nullptr) {
      return nullptr;
    }
    node = child;
  }
  return node;
}

inline void BSkipList::FixRightPrev(Node* neu, Node* right) {
  if (right == nullptr) {
    return;
  }
  // Hold only `right`. Re-walk from `neu`: a later splice may already have
  // linked a node between `neu` and `right`, and it will also take this lock.
  // The last writer to finish the walk stores the predecessor that is current
  // at that moment.
  LockNode(right);
  Node* pred = neu;
  for (;;) {
    Node* nxt = pred->next.load(std::memory_order_acquire);
    if (nxt == right || nxt == nullptr) {
      break;
    }
    pred = nxt;
  }
  if (pred->next.load(std::memory_order_acquire) == right) {
    right->prev.store(pred, std::memory_order_relaxed);
  }
  EndWrite(right);
}

inline void BSkipList::PublishTail(Node* candidate) {
  if (candidate->level != 0) {
    return;
  }
  // `tail_` is an append hint. Swing it to the rightmost node reachable from
  // `candidate`, and retry so a splice that landed during the walk cannot
  // leave the hint behind that node.
  for (int attempt = 0; attempt < 64; ++attempt) {
    Node* tip = candidate;
    for (;;) {
      Node* nxt = tip->next.load(std::memory_order_acquire);
      if (nxt == nullptr) {
        break;
      }
      tip = nxt;
    }
    Node* cur = tail_.load(std::memory_order_acquire);
    if (cur == tip &&
        tip->next.load(std::memory_order_acquire) == nullptr) {
      return;
    }
    if (tail_.compare_exchange_strong(cur, tip, std::memory_order_release,
                                      std::memory_order_acquire)) {
      candidate = tip;
      if (tip->next.load(std::memory_order_acquire) == nullptr) {
        return;
      }
    }
  }
}

inline void BSkipList::SpliceLocked(Node* left, Node* neu,
                                    uint32_t new_left_count) {
  // `left` is already locked. Publish the sibling, then fix `right->prev`
  // after releasing `left` so two splices cannot wait on each other.
  Node* right = left->next.load(std::memory_order_relaxed);
  neu->prev.store(left, std::memory_order_relaxed);
  if (right != nullptr) {
    StoreNextHeader(neu, right);
  }
  neu->next.store(right, std::memory_order_release);
  left->count.store(new_left_count, std::memory_order_relaxed);
  StoreNextHeader(left, neu);
  left->next.store(neu, std::memory_order_release);
  EndWrite(left);
  FixRightPrev(neu, right);
  if (right == nullptr) {
    PublishTail(neu);
  }
}

inline BSkipList::Node* BSkipList::PromotionSplitLocked(Node* node,
                                                        uint32_t rank,
                                                        const KeyView& key,
                                                        Node* new_child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(rank < count);
  Node* neu = NewNode(node->level);
  WriteSlot(neu, 0, key);
  if (node->level > 0) {
    neu->children[0].store(new_child, std::memory_order_relaxed);
  }
  const uint32_t move = count - (rank + 1);
  for (uint32_t i = 0; i < move; ++i) {
    CopySlot(neu, 1 + i, node, rank + 1 + i);
    if (node->level > 0) {
      Node* child = node->children[rank + 1 + i].load(std::memory_order_relaxed);
      neu->children[1 + i].store(child, std::memory_order_relaxed);
    }
  }
  neu->count.store(1 + move, std::memory_order_relaxed);
  SpliceLocked(node, neu, rank + 1);
  return neu;
}

inline void BSkipList::SplitAndInsertLocked(Node* node, uint32_t rank,
                                            const KeyView& key,
                                            Node* new_child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count == kFanout);
  assert(rank < count);
  SlotView slots[kFanout + 1];
  Node* kids[kFanout + 1];
  for (uint32_t i = 0; i < count; ++i) {
    slots[i] = ReadSlot(node, i);
    if (node->level > 0) {
      kids[i] = node->children[i].load(std::memory_order_relaxed);
    }
  }
  const uint32_t pos = rank + 1;
  for (uint32_t i = count; i > pos; --i) {
    slots[i] = slots[i - 1];
    if (node->level > 0) {
      kids[i] = kids[i - 1];
    }
  }
  slots[pos].entry = key.entry;
  slots[pos].user_len = key.user_len;
  slots[pos].footer = key.footer;
  slots[pos].word0 = key.word0;
  slots[pos].word1 = key.word1;
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    slots[pos].extra[t] = key.extra[t];
  }
  memset(slots[pos].user, 0, kInlineUserKey);
  const uint32_t ncopy =
      key.user_len < kInlineUserKey ? key.user_len : kInlineUserKey;
  if (ncopy > 0) {
    memcpy(slots[pos].user, key.user, ncopy);
  }
  if (node->level > 0) {
    kids[pos] = new_child;
  }
  const uint32_t total = count + 1;
  const uint32_t left_count = total / 2;
  const uint32_t right_count = total - left_count;
  Node* neu = NewNode(node->level);
  for (uint32_t i = 0; i < right_count; ++i) {
    WriteSlot(neu, i, slots[left_count + i]);
    if (node->level > 0) {
      neu->children[i].store(kids[left_count + i], std::memory_order_relaxed);
    }
  }
  neu->count.store(right_count, std::memory_order_relaxed);

  Node* right = node->next.load(std::memory_order_relaxed);
  neu->prev.store(node, std::memory_order_relaxed);
  if (right != nullptr) {
    StoreNextHeader(neu, right);
  }
  neu->next.store(right, std::memory_order_release);
  for (uint32_t i = 1; i < left_count; ++i) {
    WriteSlot(node, i, slots[i]);
    if (node->level > 0) {
      node->children[i].store(kids[i], std::memory_order_relaxed);
    }
  }
  node->count.store(left_count, std::memory_order_relaxed);
  StoreNextHeader(node, neu);
  node->next.store(neu, std::memory_order_release);
  EndWrite(node);
  FixRightPrev(neu, right);
  if (right == nullptr) {
    PublishTail(neu);
  }
}

inline void BSkipList::ShiftInsertLocked(Node* node, uint32_t pos,
                                         const KeyView& key, Node* child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count < kFanout);
  assert(pos > 0);
  assert(pos <= count);
  for (uint32_t i = count; i > pos; --i) {
    CopySlot(node, i, node, i - 1);
    if (node->level > 0) {
      Node* moved = node->children[i - 1].load(std::memory_order_relaxed);
      node->children[i].store(moved, std::memory_order_relaxed);
    }
  }
  WriteSlot(node, pos, key);
  if (node->level > 0) {
    node->children[pos].store(child, std::memory_order_relaxed);
  }
  node->count.store(count + 1, std::memory_order_relaxed);
  EndWrite(node);
}

inline bool BSkipList::TryAppendConcurrent(const KeyView& key) {
  Node* leaf = tail_.load(std::memory_order_acquire);
  if (leaf == nullptr) {
    return false;
  }
  // Reject without locking. Random inserts must not queue on the tail.
  const uint32_t observed = leaf->count.load(std::memory_order_acquire);
  if (observed == 0 || observed > kFanout) {
    return false;
  }
  const SlotView last = ReadSlot(leaf, observed - 1);
  if (CompareKeyToSlot(key, last) <= 0) {
    return false;
  }
  if (leaf->next.load(std::memory_order_acquire) != nullptr &&
      CompareKeyToNextHeader(key, leaf) >= 0) {
    return false;
  }
  if (!TryLockNode(leaf)) {
    return false;
  }
  const uint32_t count = leaf->count.load(std::memory_order_relaxed);
  if (count == 0 || count > kFanout || !KeyBelongs(leaf, key)) {
    EndWrite(leaf);
    return false;
  }
  const SlotView locked_last = ReadSlot(leaf, count - 1);
  if (CompareKeyToSlot(key, locked_last) <= 0) {
    EndWrite(leaf);
    return false;
  }
  if (count < kFanout) {
    WriteSlot(leaf, count, key);
    leaf->count.store(count + 1, std::memory_order_relaxed);
    EndWrite(leaf);
    return true;
  }
  constexpr uint32_t half = kFanout / 2;
  Node* neu = NewNode(0);
  const uint32_t moved = kFanout - half;
  for (uint32_t i = 0; i < moved; ++i) {
    CopySlot(neu, i, leaf, half + i);
  }
  WriteSlot(neu, moved, key);
  neu->count.store(moved + 1, std::memory_order_relaxed);
  SpliceLocked(leaf, neu, half);
  return true;
}

inline BSkipList::Node* BSkipList::PromoteConcurrent(const KeyView& key,
                                                     int level,
                                                     Node* new_child) {
  for (;;) {
    Node* node = Locate(key, level);
    if (node == nullptr || node->level != static_cast<uint32_t>(level)) {
      port::AsmVolatilePause();
      continue;
    }
    LockNode(node);
    if (!KeyBelongs(node, key)) {
      EndWrite(node);
      continue;
    }
    const uint32_t count = node->count.load(std::memory_order_relaxed);
    SlotView at;
    const uint32_t rank = FindRank(node, count, key, &at);
    if (at.entry != nullptr && CompareKeyToSlot(key, at) == 0) {
      EndWrite(node);
      return nullptr;
    }
    return PromotionSplitLocked(node, rank, key, new_child);
  }
}

inline bool BSkipList::InsertAtLevelConcurrent(const KeyView& key, int level,
                                               Node* child) {
  for (;;) {
    Node* node = Locate(key, level);
    if (node == nullptr || node->level != static_cast<uint32_t>(level)) {
      port::AsmVolatilePause();
      continue;
    }
    LockNode(node);
    if (!KeyBelongs(node, key)) {
      EndWrite(node);
      continue;
    }
    const uint32_t count = node->count.load(std::memory_order_relaxed);
    SlotView at;
    const uint32_t rank = FindRank(node, count, key, &at);
    if (at.entry != nullptr && CompareKeyToSlot(key, at) == 0) {
      EndWrite(node);
      // A lower level of this insert already linked `key`. The separator is
      // present, so the key is in the list.
      return child != nullptr;
    }
    if (count >= kFanout) {
      SplitAndInsertLocked(node, rank, key, child);
    } else {
      ShiftInsertLocked(node, rank + 1, key, child);
    }
    return true;
  }
}

inline void BSkipList::PublishHeight(int height) {
  int seen = height_.load(std::memory_order_relaxed);
  while (height + 1 > seen) {
    if (height_.compare_exchange_weak(seen, height + 1,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
      break;
    }
  }
}

inline void BSkipList::SpliceAfter(Node* left, Node* neu,
                                   uint32_t new_left_count) {
  Node* right = left->next.load(std::memory_order_relaxed);
  neu->prev.store(left, std::memory_order_relaxed);
  neu->next.store(right, std::memory_order_relaxed);
  if (right != nullptr) {
    StoreNextHeader(neu, right);
  }
  BeginWrite(left);
  left->count.store(new_left_count, std::memory_order_relaxed);
  StoreNextHeader(left, neu);
  left->next.store(neu, std::memory_order_relaxed);
  EndWrite(left);
  if (right != nullptr) {
    BeginWrite(right);
    right->prev.store(neu, std::memory_order_relaxed);
    EndWrite(right);
  }
  if (right == nullptr && left->level == 0) {
    tail_.store(neu, std::memory_order_release);
  }
}

inline bool BSkipList::TryAppend(const KeyView& key) {
  Node* leaf = tail_.load(std::memory_order_relaxed);
  const uint32_t count = leaf->count.load(std::memory_order_relaxed);
  if (count == 0 || count > kFanout) {
    return false;
  }
  const SlotView last = ReadSlot(leaf, count - 1);
  if (CompareKeyToSlot(key, last) <= 0) {
    return false;
  }
  if (leaf->next.load(std::memory_order_relaxed) != nullptr &&
      CompareKeyToNextHeader(key, leaf) >= 0) {
    return false;
  }
  if (count < kFanout) {
    BeginWrite(leaf);
    WriteSlot(leaf, count, key);
    leaf->count.store(count + 1, std::memory_order_relaxed);
    EndWrite(leaf);
    return true;
  }

  constexpr uint32_t half = kFanout / 2;
  Node* neu = NewNode(0);
  const uint32_t moved = kFanout - half;
  for (uint32_t i = 0; i < moved; ++i) {
    CopySlot(neu, i, leaf, half + i);
  }
  WriteSlot(neu, moved, key);
  neu->count.store(moved + 1, std::memory_order_relaxed);
  SpliceAfter(leaf, neu, half);
  return true;
}

inline BSkipList::Node* BSkipList::PromotionSplit(Node* node, uint32_t rank,
                                                  const KeyView& key,
                                                  Node* new_child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(rank < count);
  Node* neu = NewNode(node->level);
  WriteSlot(neu, 0, key);
  if (node->level > 0) {
    neu->children[0].store(new_child, std::memory_order_relaxed);
  }
  const uint32_t move = count - (rank + 1);
  for (uint32_t i = 0; i < move; ++i) {
    CopySlot(neu, 1 + i, node, rank + 1 + i);
    if (node->level > 0) {
      Node* child =
          node->children[rank + 1 + i].load(std::memory_order_relaxed);
      neu->children[1 + i].store(child, std::memory_order_relaxed);
    }
  }
  neu->count.store(1 + move, std::memory_order_relaxed);
  SpliceAfter(node, neu, rank + 1);
  return neu;
}

inline void BSkipList::SplitAndInsert(Node* node, uint32_t rank,
                                      const KeyView& key, Node* new_child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count == kFanout);
  assert(rank < count);
  SlotView slots[kFanout + 1];
  Node* kids[kFanout + 1];
  for (uint32_t i = 0; i < count; ++i) {
    slots[i] = ReadSlot(node, i);
    if (node->level > 0) {
      kids[i] = node->children[i].load(std::memory_order_relaxed);
    }
  }
  const uint32_t pos = rank + 1;
  for (uint32_t i = count; i > pos; --i) {
    slots[i] = slots[i - 1];
    if (node->level > 0) {
      kids[i] = kids[i - 1];
    }
  }
  slots[pos].entry = key.entry;
  slots[pos].user_len = key.user_len;
  slots[pos].footer = key.footer;
  slots[pos].word0 = key.word0;
  slots[pos].word1 = key.word1;
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    slots[pos].extra[t] = key.extra[t];
  }
  memset(slots[pos].user, 0, kInlineUserKey);
  const uint32_t ncopy =
      key.user_len < kInlineUserKey ? key.user_len : kInlineUserKey;
  if (ncopy > 0) {
    memcpy(slots[pos].user, key.user, ncopy);
  }
  if (node->level > 0) {
    kids[pos] = new_child;
  }
  const uint32_t total = count + 1;
  const uint32_t left_count = total / 2;
  const uint32_t right_count = total - left_count;
  Node* neu = NewNode(node->level);
  for (uint32_t i = 0; i < right_count; ++i) {
    WriteSlot(neu, i, slots[left_count + i]);
    if (node->level > 0) {
      neu->children[i].store(kids[left_count + i], std::memory_order_relaxed);
    }
  }
  neu->count.store(right_count, std::memory_order_relaxed);

  Node* right = node->next.load(std::memory_order_relaxed);
  neu->prev.store(node, std::memory_order_relaxed);
  neu->next.store(right, std::memory_order_relaxed);
  if (right != nullptr) {
    StoreNextHeader(neu, right);
  }
  BeginWrite(node);
  for (uint32_t i = 1; i < left_count; ++i) {
    WriteSlot(node, i, slots[i]);
    if (node->level > 0) {
      node->children[i].store(kids[i], std::memory_order_relaxed);
    }
  }
  node->count.store(left_count, std::memory_order_relaxed);
  StoreNextHeader(node, neu);
  node->next.store(neu, std::memory_order_relaxed);
  EndWrite(node);
  if (right != nullptr) {
    BeginWrite(right);
    right->prev.store(neu, std::memory_order_relaxed);
    EndWrite(right);
  }
  if (right == nullptr && node->level == 0) {
    tail_.store(neu, std::memory_order_release);
  }
}

inline void BSkipList::ShiftInsert(Node* node, uint32_t pos, const KeyView& key,
                                   Node* child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count < kFanout);
  assert(pos > 0);
  assert(pos <= count);
  BeginWrite(node);
  for (uint32_t i = count; i > pos; --i) {
    CopySlot(node, i, node, i - 1);
    if (node->level > 0) {
      Node* moved = node->children[i - 1].load(std::memory_order_relaxed);
      node->children[i].store(moved, std::memory_order_relaxed);
    }
  }
  WriteSlot(node, pos, key);
  if (node->level > 0) {
    node->children[pos].store(child, std::memory_order_relaxed);
  }
  node->count.store(count + 1, std::memory_order_relaxed);
  EndWrite(node);
}

inline bool BSkipList::Insert(const char* entry) {
  const KeyView key = ParseEntry(entry);
  const int height = RandomHeight();
  if (LIKELY(height == 0) && TryAppend(key)) {
    num_entries_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  Step path[kMaxHeight];
  const int top = height_.load(std::memory_order_relaxed);
  assert(top >= 1 && top <= kMaxHeight);
  // Levels above `top` hold only the sentinel. Record them without walking,
  // and start the real search at the highest built level.
  const int start = height > top - 1 ? height : top - 1;
  Node* curr = nullptr;
  for (int level = start; level >= 0; --level) {
    if (level >= top) {
      path[level].node = headers_[level];
      path[level].rank = 0;
      continue;
    }
    if (curr == nullptr) {
      curr = headers_[level];
    }
    curr = WalkRight(curr, key);
    SlotView at;
    const uint32_t count = curr->count.load(std::memory_order_relaxed);
    const uint32_t rank = FindRank(curr, count, key, &at);
    if (at.entry != nullptr && CompareKeyToSlot(key, at) == 0) {
      return false;
    }
    path[level].node = curr;
    path[level].rank = rank;
    if (level > 0) {
      curr = curr->children[rank].load(std::memory_order_relaxed);
      assert(curr != nullptr);
    }
  }

  Node* below = nullptr;
  for (int level = 0; level < height; ++level) {
    below = PromotionSplit(path[level].node, path[level].rank, key, below);
  }
  Node* node = path[height].node;
  const uint32_t rank = path[height].rank;
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  if (count >= kFanout) {
    SplitAndInsert(node, rank, key, below);
  } else {
    ShiftInsert(node, rank + 1, key, below);
  }
  if (height + 1 > top) {
    height_.store(height + 1, std::memory_order_release);
  }
  num_entries_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline bool BSkipList::InsertConcurrently(const char* entry) {
  const KeyView key = ParseEntry(entry);
  const int height = RandomHeight();
  if (LIKELY(height == 0) && TryAppendConcurrent(key)) {
    num_entries_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  Node* below = nullptr;
  for (int level = 0; level < height; ++level) {
    Node* created = PromoteConcurrent(key, level, below);
    if (created == nullptr) {
      if (level == 0) {
        return false;
      }
      // The key is already linked at a lower level.
      PublishHeight(height);
      num_entries_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    below = created;
  }
  if (!InsertAtLevelConcurrent(key, height, below)) {
    return false;
  }
  PublishHeight(height);
  num_entries_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

inline bool BSkipList::Descend(const KeyView& key, Node** leaf,
                               uint32_t* index, bool* equal) const {
  for (;;) {
    const int top = height_.load(std::memory_order_acquire);
    const int start = top > 0 ? top - 1 : 0;
    Node* node = headers_[start];
    bool failed = false;
    for (int level = start; level >= 0; --level) {
      for (;;) {
        const uint32_t seq1 = node->seq.load(std::memory_order_acquire);
        if (UNLIKELY((seq1 & 1u) != 0)) {
          port::AsmVolatilePause();
          failed = true;
          break;
        }
        Node* next = node->next.load(std::memory_order_relaxed);
        const bool go_right =
            next != nullptr && CompareKeyToNextHeader(key, node) >= 0;
        uint32_t count = 0;
        uint32_t rank = 0;
        SlotView at;
        Node* child = nullptr;
        if (!go_right) {
          count = node->count.load(std::memory_order_relaxed);
          if (count == 0 || count > kFanout) {
            failed = true;
            break;
          }
          rank = FindRank(node, count, key, &at);
          if (node->level > 0) {
            child = node->children[rank].load(std::memory_order_relaxed);
          }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (node->seq.load(std::memory_order_relaxed) != seq1) {
          failed = true;
          break;
        }
        if (go_right) {
          node = next;
          continue;
        }
        if (level == 0) {
          *leaf = node;
          *index = rank;
          *equal = at.entry != nullptr && CompareKeyToSlot(key, at) == 0;
          return true;
        }
        if (child == nullptr) {
          failed = true;
          break;
        }
        node = child;
        break;
      }
      if (failed) {
        break;
      }
    }
    if (!failed) {
      return true;
    }
  }
}

inline bool BSkipList::Contains(const char* key) const {
  const KeyView view = ParseEntry(key);
  Node* leaf = nullptr;
  uint32_t index = 0;
  bool equal = false;
  Descend(view, &leaf, &index, &equal);
  return equal;
}

inline bool BSkipList::ReadEntryAt(Node* node, uint32_t index,
                                   const char** entry, Node** next) const {
  const uint32_t seq1 = node->seq.load(std::memory_order_acquire);
  if ((seq1 & 1u) != 0) {
    return false;
  }
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  const char* got = nullptr;
  if (index < count && count <= kFanout) {
    got = node->slots[index].entry.load(std::memory_order_relaxed);
  }
  Node* nxt = node->next.load(std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_acquire);
  if (node->seq.load(std::memory_order_relaxed) != seq1) {
    return false;
  }
  *entry = got;
  *next = nxt;
  return true;
}

inline uint64_t BSkipList::ApproximateNumEntries(const Slice& start_ikey,
                                                 const Slice& end_ikey) const {
  if (start_ikey.size() < 8 || end_ikey.size() < 8) {
    return 0;
  }
  const KeyView start = ParseInternalKey(start_ikey);
  const KeyView end = ParseInternalKey(end_ikey);
  if (CompareFull(start.user, start.user_len, start.footer, end.user,
                  end.user_len, end.footer) >= 0) {
    return 0;
  }
  const int top = height_.load(std::memory_order_acquire);
  for (int level = top > 0 ? top - 1 : 0; level >= 0; --level) {
    uint64_t local = 0;
    Node* node = headers_[level];
    node = WalkRight(node, start);
    while (node != nullptr) {
      const uint32_t count = node->count.load(std::memory_order_relaxed);
      bool stop = false;
      for (uint32_t i = 0; i < count && i < kFanout; ++i) {
        const SlotView slot = ReadSlot(node, i);
        if (slot.entry == nullptr) {
          continue;
        }
        if (CompareKeyToSlot(start, slot) > 0) {
          continue;
        }
        if (CompareKeyToSlot(end, slot) <= 0) {
          stop = true;
          break;
        }
        ++local;
      }
      if (stop) {
        break;
      }
      Node* next = node->next.load(std::memory_order_relaxed);
      if (next == nullptr) {
        break;
      }
      if (CompareKeyToNextHeader(end, node) <= 0) {
        break;
      }
      node = next;
    }
    if (local >= 8 || level == 0) {
      uint64_t scaled = local;
      for (int i = 0; i < level; ++i) {
        scaled *= kBranch;
      }
      return scaled;
    }
  }
  return 0;
}

inline void BSkipList::UniqueRandomSample(
    uint64_t target_sample_size,
    std::unordered_set<const char*>* entries) const {
  assert(target_sample_size > 0);
  entries->clear();
  std::vector<const char*> reservoir;
  reservoir.reserve(static_cast<size_t>(target_sample_size));
  uint64_t seen = 0;
  Random* rnd = Random::GetTLSInstance();
  for (Node* node = headers_[0]; node != nullptr;
       node = node->next.load(std::memory_order_acquire)) {
    const uint32_t count = node->count.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < count && i < kFanout; ++i) {
      const char* entry = node->slots[i].entry.load(std::memory_order_relaxed);
      if (entry == nullptr) {
        continue;
      }
      ++seen;
      if (reservoir.size() < target_sample_size) {
        reservoir.push_back(entry);
      } else {
        const uint64_t pick = rnd->Next64() % seen;
        if (pick < target_sample_size) {
          reservoir[static_cast<size_t>(pick)] = entry;
        }
      }
    }
  }
  entries->insert(reservoir.begin(), reservoir.end());
}

inline void BSkipList::TEST_Validate() const {
  for (int level = 0; level < kMaxHeight; ++level) {
    const Node* node = headers_[level];
    const Node* prev = nullptr;
    while (node != nullptr) {
      const uint32_t count = node->count.load(std::memory_order_relaxed);
      assert(count >= 1);
      assert(count <= kFanout);
      assert(node->level == static_cast<uint32_t>(level));
      assert(node->prev.load(std::memory_order_relaxed) == prev);
      SlotView previous;
      bool have_previous = false;
      for (uint32_t i = 0; i < count; ++i) {
        const SlotView slot = ReadSlot(node, i);
        if (i == 0 && prev == nullptr) {
          assert(slot.entry == nullptr);
        }
        if (slot.entry != nullptr && have_previous &&
            previous.entry != nullptr) {
          assert(CompareFull(previous.user_len <= kInlineUserKey
                                 ? previous.user
                                 : FullUser(previous),
                             previous.user_len, previous.footer,
                             slot.user_len <= kInlineUserKey ? slot.user
                                                             : FullUser(slot),
                             slot.user_len, slot.footer) < 0);
        }
        if (slot.entry != nullptr) {
          previous = slot;
          have_previous = true;
        }
        if (level > 0) {
          Node* child = node->children[i].load(std::memory_order_relaxed);
          assert(child != nullptr);
          const SlotView child_head = ReadSlot(child, 0);
          if (slot.entry == nullptr) {
            assert(child_head.entry == nullptr);
          } else if (child_head.entry != nullptr) {
            assert(CompareFull(slot.user_len <= kInlineUserKey ? slot.user
                                                               : FullUser(slot),
                               slot.user_len, slot.footer,
                               child_head.user_len <= kInlineUserKey
                                   ? child_head.user
                                   : FullUser(child_head),
                               child_head.user_len, child_head.footer) == 0);
          }
        }
      }
      Node* next = node->next.load(std::memory_order_relaxed);
#ifndef NDEBUG
      if (next != nullptr) {
        const SlotView next_head = ReadSlot(next, 0);
        assert(next_head.entry != nullptr);
        KeyView probe;
        probe.entry = next_head.entry;
        probe.user = next_head.user_len <= kInlineUserKey ? next_head.user
                                                          : FullUser(next_head);
        probe.user_len = next_head.user_len;
        probe.footer = next_head.footer;
        probe.word0 = next_head.word0;
        probe.word1 = next_head.word1;
        for (uint32_t t = 0; t < kExtraWords; ++t) {
          probe.extra[t] = next_head.extra[t];
        }
        assert(CompareKeyToNextHeader(probe, node) == 0);
      }
#endif
      prev = node;
      node = next;
    }
  }
}

inline BSkipList::Iterator::Iterator(const BSkipList* list)
    : list_(list),
      leaf_(nullptr),
      index_(0),
      key_(nullptr),
      cached_leaf_(nullptr),
      cached_next_(nullptr),
      cached_seq_(0),
      cached_count_(0) {}

inline void BSkipList::Iterator::Remember(Node* leaf, uint32_t seq,
                                          uint32_t count, Node* next) {
  cached_leaf_ = leaf;
  cached_next_ = next;
  cached_seq_ = seq;
  cached_count_ = count;
}

inline void BSkipList::Iterator::Seek(const char* target) {
  cached_leaf_ = nullptr;
  const KeyView key = list_->ParseEntry(target);
  for (;;) {
    Node* leaf = nullptr;
    uint32_t rank = 0;
    bool equal = false;
    list_->Descend(key, &leaf, &rank, &equal);
    if (equal) {
      const char* entry = nullptr;
      Node* next = nullptr;
      if (!list_->ReadEntryAt(leaf, rank, &entry, &next) || entry == nullptr) {
        continue;
      }
      const SlotView slot = list_->ReadSlot(leaf, rank);
      std::atomic_thread_fence(std::memory_order_acquire);
      if (list_->CompareKeyToSlot(key, slot) != 0 || slot.entry == nullptr) {
        continue;
      }
      leaf_ = leaf;
      index_ = rank;
      key_ = slot.entry;
      return;
    }
    // `rank` is the rightmost slot <= key, so the ceiling is the next entry.
    uint32_t index = rank + 1;
    bool restarted = false;
    while (!restarted) {
      const char* entry = nullptr;
      Node* next = nullptr;
      if (!list_->ReadEntryAt(leaf, index, &entry, &next)) {
        restarted = true;
        break;
      }
      if (entry != nullptr) {
        leaf_ = leaf;
        index_ = index;
        key_ = entry;
        return;
      }
      const uint32_t count = leaf->count.load(std::memory_order_acquire);
      if (index + 1 < count && count <= kFanout) {
        ++index;
        continue;
      }
      if (next == nullptr) {
        leaf_ = nullptr;
        key_ = nullptr;
        return;
      }
      leaf = next;
      index = 0;
    }
  }
}

inline void BSkipList::Iterator::SeekForPrev(const char* target) {
  cached_leaf_ = nullptr;
  Seek(target);
  if (!Valid()) {
    SeekToLast();
    return;
  }
  const KeyView key = list_->ParseEntry(target);
  const SlotView slot = list_->ReadSlot(leaf_, index_);
  if (slot.entry == key_ && list_->CompareKeyToSlot(key, slot) == 0) {
    return;
  }
  Prev();
}

inline void BSkipList::Iterator::SeekToFirst() {
  cached_leaf_ = nullptr;
  Node* leaf = list_->headers_[0];
  uint32_t index = 0;
  for (;;) {
    const char* entry = nullptr;
    Node* next = nullptr;
    if (!list_->ReadEntryAt(leaf, index, &entry, &next)) {
      port::AsmVolatilePause();
      continue;
    }
    if (entry != nullptr) {
      leaf_ = leaf;
      index_ = index;
      key_ = entry;
      return;
    }
    const uint32_t count = leaf->count.load(std::memory_order_acquire);
    if (index + 1 < count) {
      ++index;
      continue;
    }
    if (next == nullptr) {
      leaf_ = nullptr;
      key_ = nullptr;
      return;
    }
    leaf = next;
    index = 0;
  }
}

inline void BSkipList::Iterator::SeekToLast() {
  cached_leaf_ = nullptr;
  Node* leaf = list_->tail_.load(std::memory_order_acquire);
  if (leaf == nullptr) {
    leaf = list_->headers_[0];
  }
  for (;;) {
    Node* next = leaf->next.load(std::memory_order_acquire);
    if (next == nullptr) {
      break;
    }
    leaf = next;
  }
  for (;;) {
    const uint32_t seq1 = leaf->seq.load(std::memory_order_acquire);
    if ((seq1 & 1u) != 0) {
      port::AsmVolatilePause();
      continue;
    }
    const uint32_t count = leaf->count.load(std::memory_order_relaxed);
    if (count == 0 || count > kFanout) {
      continue;
    }
    const char* entry =
        leaf->slots[count - 1].entry.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (leaf->seq.load(std::memory_order_relaxed) != seq1) {
      continue;
    }
    if (entry == nullptr) {
      leaf_ = nullptr;
      key_ = nullptr;
      return;
    }
    leaf_ = leaf;
    index_ = count - 1;
    key_ = entry;
    return;
  }
}

inline void BSkipList::Iterator::Next() {
  const char* current = key_;
  assert(current != nullptr);
  if (LIKELY(leaf_ == cached_leaf_)) {
    const uint32_t seq = leaf_->seq.load(std::memory_order_acquire);
    if (LIKELY(seq == cached_seq_)) {
      if (index_ + 1 < cached_count_) {
        const char* nxt =
            leaf_->slots[index_ + 1].entry.load(std::memory_order_relaxed);
        if (LIKELY(nxt != nullptr)) {
          ++index_;
          key_ = nxt;
          return;
        }
      } else if (cached_next_ == nullptr) {
        leaf_ = nullptr;
        key_ = nullptr;
        cached_leaf_ = nullptr;
        return;
      } else {
        const char* nxt =
            cached_next_->slots[0].entry.load(std::memory_order_relaxed);
        if (LIKELY(nxt != nullptr)) {
          leaf_ = cached_next_;
          index_ = 0;
          key_ = nxt;
          cached_leaf_ = nullptr;
          return;
        }
      }
    }
  }
  for (;;) {
    const uint32_t seq1 = leaf_->seq.load(std::memory_order_acquire);
    if ((seq1 & 1u) != 0) {
      port::AsmVolatilePause();
      continue;
    }
    const uint32_t count = leaf_->count.load(std::memory_order_relaxed);
    const char* at =
        index_ < count && count <= kFanout
            ? leaf_->slots[index_].entry.load(std::memory_order_relaxed)
            : nullptr;
    Node* next = leaf_->next.load(std::memory_order_relaxed);
    const char* advanced = nullptr;
    Node* advanced_leaf = nullptr;
    uint32_t advanced_index = 0;
    bool moved = false;
    bool finished = false;
    if (at == current) {
      if (index_ + 1 < count) {
        advanced = leaf_->slots[index_ + 1].entry.load(std::memory_order_relaxed);
        advanced_leaf = leaf_;
        advanced_index = index_ + 1;
        moved = true;
      } else if (next != nullptr) {
        advanced = next->slots[0].entry.load(std::memory_order_relaxed);
        advanced_leaf = next;
        advanced_index = 0;
        moved = true;
      } else {
        finished = true;
      }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (leaf_->seq.load(std::memory_order_relaxed) != seq1) {
      continue;
    }
    if (finished) {
      leaf_ = nullptr;
      key_ = nullptr;
      return;
    }
    if (moved) {
      const bool same_leaf = advanced_leaf == leaf_;
      leaf_ = advanced_leaf;
      index_ = advanced_index;
      if (advanced == nullptr) {
        // Sentinel, or a slot that moved. Step again from this position.
        key_ = nullptr;
        const char* entry = nullptr;
        Node* nxt = nullptr;
        if (!list_->ReadEntryAt(leaf_, index_, &entry, &nxt)) {
          continue;
        }
        if (entry != nullptr) {
          key_ = entry;
          return;
        }
        if (index_ + 1 < leaf_->count.load(std::memory_order_acquire)) {
          ++index_;
          continue;
        }
        if (nxt == nullptr) {
          leaf_ = nullptr;
          key_ = nullptr;
          return;
        }
        leaf_ = nxt;
        index_ = 0;
        continue;
      }
      key_ = advanced;
      if (same_leaf) {
        Remember(leaf_, seq1, count, next);
      } else {
        cached_leaf_ = nullptr;
      }
      return;
    }
    Seek(current);
    if (key_ != current) {
      return;
    }
  }
}

inline void BSkipList::Iterator::Prev() {
  cached_leaf_ = nullptr;
  const char* current = key_;
  assert(current != nullptr);
  for (int attempt = 0; attempt < 8; ++attempt) {
    const uint32_t seq1 = leaf_->seq.load(std::memory_order_acquire);
    if ((seq1 & 1u) != 0) {
      port::AsmVolatilePause();
      continue;
    }
    const uint32_t count = leaf_->count.load(std::memory_order_relaxed);
    const char* at =
        index_ < count && count <= kFanout
            ? leaf_->slots[index_].entry.load(std::memory_order_relaxed)
            : nullptr;
    const char* prev_entry = nullptr;
    uint32_t prev_index = 0;
    Node* prev_leaf = nullptr;
    bool local = false;
    bool use_prev_node = false;
    if (at == current && index_ > 0) {
      prev_entry =
          leaf_->slots[index_ - 1].entry.load(std::memory_order_relaxed);
      prev_index = index_ - 1;
      prev_leaf = leaf_;
      local = true;
    } else if (at == current && index_ == 0) {
      prev_leaf = leaf_->prev.load(std::memory_order_relaxed);
      use_prev_node = prev_leaf != nullptr;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (leaf_->seq.load(std::memory_order_relaxed) != seq1) {
      continue;
    }
    if (local) {
      if (prev_entry == nullptr) {
        leaf_ = nullptr;
        key_ = nullptr;
        return;
      }
      index_ = prev_index;
      key_ = prev_entry;
      return;
    }
    if (use_prev_node) {
      Node* back = prev_leaf->next.load(std::memory_order_acquire);
      if (back != leaf_) {
        break;
      }
      const uint32_t prev_count =
          prev_leaf->count.load(std::memory_order_acquire);
      if (prev_count == 0 || prev_count > kFanout) {
        continue;
      }
      const char* entry = prev_leaf->slots[prev_count - 1].entry.load(
          std::memory_order_relaxed);
      if (entry == nullptr) {
        leaf_ = nullptr;
        key_ = nullptr;
        return;
      }
      leaf_ = prev_leaf;
      index_ = prev_count - 1;
      key_ = entry;
      return;
    }
    break;
  }
  // The node changed under us. Land on `current` and step back by a search
  // for the previous key: Seek positions at the first entry >= current.
  Seek(current);
  if (!Valid()) {
    return;
  }
  if (key_ != current) {
    // Seek stopped on a successor, which is not a predecessor.
    // Walk back one step only when it is still local; otherwise scan.
  }
  if (key_ == current && index_ > 0) {
    const char* entry = nullptr;
    Node* next = nullptr;
    if (list_->ReadEntryAt(leaf_, index_ - 1, &entry, &next) &&
        entry != nullptr) {
      --index_;
      key_ = entry;
      return;
    }
  }
  if (key_ == current) {
    Node* prev = leaf_->prev.load(std::memory_order_acquire);
    if (prev == nullptr) {
      leaf_ = nullptr;
      key_ = nullptr;
      return;
    }
    const uint32_t prev_count = prev->count.load(std::memory_order_acquire);
    if (prev_count == 0) {
      leaf_ = nullptr;
      key_ = nullptr;
      return;
    }
    const char* entry =
        prev->slots[prev_count - 1].entry.load(std::memory_order_relaxed);
    if (entry == nullptr) {
      leaf_ = nullptr;
      key_ = nullptr;
      return;
    }
    leaf_ = prev;
    index_ = prev_count - 1;
    key_ = entry;
    return;
  }
  // Seek landed strictly after `current` only when `current` disappeared,
  // which does not happen: entries are not removed. Treat the landed key as
  // already past `current` and step one back by scanning from the start if
  // the local prev link was stale.
  const char* landed = key_;
  SeekToFirst();
  const char* last = nullptr;
  Node* last_leaf = nullptr;
  uint32_t last_index = 0;
  while (Valid() && key_ != landed) {
    last = key_;
    last_leaf = leaf_;
    last_index = index_;
    Next();
  }
  if (last == nullptr) {
    leaf_ = nullptr;
    key_ = nullptr;
    return;
  }
  leaf_ = last_leaf;
  index_ = last_index;
  key_ = last;
}

}  // namespace ROCKSDB_NAMESPACE
