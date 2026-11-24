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
// The seqlock stays odd only while a writer copies slots and swings
// pointers. Allocation takes the arena lock and can page-fault; doing that
// with the seqlock held, while other writers pause-spin on the same node,
// keeps the owner off-CPU and the write stalls.
//
// Ordering is bytewise on the user key, then descending internal-key footer
// (sequence and type). That matches InternalKeyComparator with
// BytewiseComparator. The caller must refuse any other comparator.
//
// The first 16 bytes of the user key and the footer live in the node. Keys
// of that size, which is db_bench's default, are compared without touching
// the memtable entry. Each node also keeps the first word of every slot in a
// contiguous array, so the rank scan reads two cache lines instead of one
// line per slot; the remaining fields stay packed as a slot payload so
// shifting and splitting move one contiguous region.

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <thread>
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

  // `hint` is private to one writer. The first call allocates it. A later key
  // that still belongs in the hinted leaf, or in one of the next few leaves,
  // is inserted there instead of searching from the root. Sequential ranges
  // that are not the list tail use this; the tail fast path does not.
  // Single-writer hints are arena-allocated. Concurrent hints are `new
  // char[]` so the memtable inserter can `delete[]` them.
  bool InsertWithHint(const char* key, void** hint);
  bool InsertWithHintConcurrently(const char* key, void** hint);

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
    // word0 lives in a separate contiguous array on Node: FindRank scans it
    // for every slot, so keeping those words together cuts a full node scan
    // from ~10 cache lines to 2.
    std::atomic<uint64_t> word1;
    std::atomic<uint64_t> footer;
    std::atomic<uint32_t> user_len;
    std::atomic<const char*> entry;

    Slot() : word1(0), footer(0), user_len(0), entry(nullptr) {}
  };

  struct alignas(64) Node {
    std::atomic<uint32_t> seq;
    std::atomic<uint32_t> count;
    // Bytes [16, 48) of each slot, `kFanout * kExtraWords` words. Null when
    // every key in the node is at most 16 bytes, so short-key nodes stay the
    // same size and the word0 scan keeps its stride. This pointer is in the
    // first cache line because the rank scan checks it while walking slots.
    std::atomic<std::atomic<uint64_t>*> extra;
    std::atomic<Node*> next;
    // Immutable after the node is published. Null on leaves. Elements are
    // mutated only inside the seqlock.
    std::atomic<Node*>* children;
    std::atomic<uint32_t> next_user_len;
    // Immutable after the node is published.
    const uint32_t level;
    std::atomic<uint64_t> next_footer;
    std::atomic<uint64_t> next_word0;
    std::atomic<uint64_t> next_word1;
    // First word of every key, packed and byte-swapped. Contiguous so the
    // rank scan reads two cache lines instead of one line per slot.
    std::atomic<uint64_t> word0[kFanout];
    // Remaining slot fields start on the next cache line.
    Slot slots[kFanout];
    std::atomic<uint64_t> next_extra[kExtraWords];
    // Only reverse iteration and splice fixup touch `prev`; keep it out of
    // the line that every search reads.
    std::atomic<Node*> prev;

    explicit Node(uint32_t lvl)
        : seq(0),
          count(0),
          extra(nullptr),
          next(nullptr),
          children(nullptr),
          next_user_len(0),
          level(lvl),
          next_footer(0),
          next_word0(0),
          next_word1(0),
          prev(nullptr) {
      for (uint32_t i = 0; i < kFanout; ++i) {
        word0[i].store(0, std::memory_order_relaxed);
      }
      for (uint32_t i = 0; i < kExtraWords; ++i) {
        next_extra[i].store(0, std::memory_order_relaxed);
      }
    }
  };

  static_assert(sizeof(Node) % alignof(std::atomic<Node*>) == 0,
                "children array must be aligned");
  static_assert(offsetof(Node, word0) == 64,
                "word0 must start on the cache line after the seqlock");
  static_assert(offsetof(Node, slots) == 64 + kFanout * sizeof(uint64_t),
                "slots must follow the contiguous word0 array");
  static_assert(sizeof(Node) % 64 == 0,
                "node size is a multiple of a cache line");

  struct Step {
    Node* node;
    uint32_t rank;
  };

  struct Rank {
    uint32_t index;
    bool equal;
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
  void CopySlots(Node* dst, uint32_t dst_index, const Node* src,
                 uint32_t src_index, uint32_t n) const;
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
  int CompareKeyToSlotAt(const KeyView& key, const Node* node,
                         uint32_t index) const;
  int CompareKeyToNextHeader(const KeyView& key, const Node* node) const;
  void StoreNextHeader(Node* node, const Node* next_node);
  Rank FindRank(const Node* node, uint32_t count, const KeyView& key) const;
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
  // `neu` is allocated by the caller before the seqlock is taken.
  Node* PromotionSplitLocked(Node* node, uint32_t rank, const KeyView& key,
                             Node* new_child, Node* neu);
  // Returns the node that holds `key` after the split.
  Node* SplitAndInsertLocked(Node* node, uint32_t rank, const KeyView& key,
                             Node* new_child, Node* neu);
  void ShiftSlotsRight(Node* node, uint32_t pos, uint32_t count);
  void ShiftInsertLocked(Node* node, uint32_t pos, const KeyView& key,
                         Node* child);
  // `landed` receives the leaf that holds the new key. False means the tail
  // path did not insert it (including a duplicate, which the slow path then
  // rejects).
  bool TryAppendConcurrent(const KeyView& key, Node** landed);
  // Null when `key` is already present at this level.
  Node* PromoteConcurrent(const KeyView& key, int level, Node* new_child);
  bool InsertAtLevelConcurrent(const KeyView& key, int level, Node* child,
                               Node** landed);
  void PublishHeight(int height);
  enum class ApplyResult { kMiss, kInserted, kDuplicate };
  struct InsertHint {
    Node* leaf;
  };
  static constexpr uint32_t kSplitLeft = (kFanout + 1) / 2;
  static constexpr uint32_t kSplitRight = kFanout + 1 - kSplitLeft;
  InsertHint* PrepareHint(void** hint, bool on_heap);
  Node* FollowHint(Node* start, const KeyView& key) const;
  // Writes the right sibling only. True when `key` itself was written there.
  bool FillSplitRight(Node* node, uint32_t pos, const KeyView& key,
                      Node* new_child, Node* neu) const;
  void PatchSplitLeft(Node* node, uint32_t pos, const KeyView& key,
                      Node* new_child);
  // Strict append into a non-full leaf. Kept separate so the common tail
  // path stays a straight line; the split and in-leaf shift are heavier.
  ApplyResult AppendInLeaf(Node* leaf, const KeyView& key, Node** landed);
  ApplyResult InsertIntoLeaf(Node* leaf, const KeyView& key, Node** landed);
  ApplyResult InsertIntoLeafConcurrent(Node* leaf, const KeyView& key,
                                       Node** landed);
  bool InsertPrepared(const KeyView& key, int height, Node** remember,
                      bool try_tail = true);
  bool InsertConcurrentPrepared(const KeyView& key, int height,
                                Node** remember);
  ApplyResult TryAppend(const KeyView& key, Node** landed);
  void SpliceAfter(Node* left, Node* neu, uint32_t new_left_count);
  Node* PromotionSplit(Node* node, uint32_t rank, const KeyView& key,
                       Node* new_child);
  Node* SplitAndInsert(Node* node, uint32_t rank, const KeyView& key,
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
  // Arena::AllocateAligned aligns to max_align_t. The seqlock and the cached
  // separator occupy the first 64 bytes only when the node itself is aligned.
  constexpr size_t kLine = 64;
  char* mem = allocator_->AllocateAligned(bytes + kLine - 1);
  const uintptr_t addr = reinterpret_cast<uintptr_t>(mem);
  char* placed = reinterpret_cast<char*>((addr + (kLine - 1)) & ~(kLine - 1));
  Node* node = new (placed) Node(level);
  if (level > 0) {
    // Construct each pointer in place. Array placement-new is not portable:
    // some standard libraries store a size cookie in front of the array.
    void* raw = placed + sizeof(Node);
    node->children = static_cast<std::atomic<Node*>*>(raw);
    for (uint32_t i = 0; i < kFanout; ++i) {
      new (node->children + i) std::atomic<Node*>(nullptr);
    }
  }
  return node;
}

inline void BSkipList::WriteSentinel(Node* node) {
  node->word0[0].store(0, std::memory_order_relaxed);
  node->slots[0].entry.store(nullptr, std::memory_order_relaxed);
  node->slots[0].footer.store(0, std::memory_order_relaxed);
  node->slots[0].user_len.store(0, std::memory_order_relaxed);
  node->slots[0].word1.store(0, std::memory_order_relaxed);
  node->count.store(1, std::memory_order_relaxed);
}

inline void BSkipList::EnsureExtra(Node* node) const {
  if (node->extra.load(std::memory_order_acquire) != nullptr) {
    return;
  }
  const size_t n = static_cast<size_t>(kFanout) * kExtraWords;
  char* raw = allocator_->AllocateAligned(n * sizeof(std::atomic<uint64_t>));
  std::atomic<uint64_t>* words =
      static_cast<std::atomic<uint64_t>*>(static_cast<void*>(raw));
  for (size_t i = 0; i < n; ++i) {
    new (words + i) std::atomic<uint64_t>(0);
  }
  std::atomic<uint64_t>* expected = nullptr;
  // Lost the race: the arena never frees, so the unused array dies with the
  // memtable. Callers must not be holding this node's seqlock; allocation
  // here is the stall the concurrent insert path avoids.
  node->extra.compare_exchange_strong(expected, words, std::memory_order_release,
                                      std::memory_order_acquire);
}

inline void BSkipList::WriteSlot(Node* node, uint32_t index,
                                 const KeyView& key) const {
  Slot* dst = &node->slots[index];
  dst->entry.store(key.entry, std::memory_order_relaxed);
  dst->footer.store(key.footer, std::memory_order_relaxed);
  dst->user_len.store(key.user_len, std::memory_order_relaxed);
  StoreUserWords(&node->word0[index], &dst->word1, key.user, key.user_len);
  if (key.user_len <= kInlineUserKey &&
      node->extra.load(std::memory_order_relaxed) == nullptr) {
    return;
  }
  EnsureExtra(node);
  std::atomic<uint64_t>* row =
      node->extra.load(std::memory_order_relaxed) + index * kExtraWords;
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    row[t].store(key.extra[t], std::memory_order_relaxed);
  }
}

inline void BSkipList::WriteSlot(Node* node, uint32_t index,
                                 const SlotView& src) const {
  Slot* dst = &node->slots[index];
  node->word0[index].store(src.word0, std::memory_order_relaxed);
  dst->word1.store(src.word1, std::memory_order_relaxed);
  dst->footer.store(src.footer, std::memory_order_relaxed);
  dst->user_len.store(src.user_len, std::memory_order_relaxed);
  dst->entry.store(src.entry, std::memory_order_relaxed);
  if (src.user_len <= kInlineUserKey &&
      node->extra.load(std::memory_order_relaxed) == nullptr) {
    return;
  }
  EnsureExtra(node);
  std::atomic<uint64_t>* row =
      node->extra.load(std::memory_order_relaxed) + index * kExtraWords;
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    row[t].store(src.extra[t], std::memory_order_relaxed);
  }
}

inline void BSkipList::CopySlots(Node* dst, uint32_t dst_index, const Node* src,
                                 uint32_t src_index, uint32_t n) const {
  // Field-major copy of a contiguous slot range. Used by splits, where the
  // source and destination ranges are both contiguous.
  for (uint32_t i = 0; i < n; ++i) {
    dst->word0[dst_index + i].store(
        src->word0[src_index + i].load(std::memory_order_relaxed),
        std::memory_order_relaxed);
  }
  for (uint32_t i = 0; i < n; ++i) {
    Slot& d = dst->slots[dst_index + i];
    const Slot& s = src->slots[src_index + i];
    d.word1.store(s.word1.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
    d.footer.store(s.footer.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
    d.user_len.store(s.user_len.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    d.entry.store(s.entry.load(std::memory_order_relaxed),
                  std::memory_order_relaxed);
  }
  if (dst->level > 0) {
    for (uint32_t i = 0; i < n; ++i) {
      dst->children[dst_index + i].store(
          src->children[src_index + i].load(std::memory_order_relaxed),
          std::memory_order_relaxed);
    }
  }
  const std::atomic<uint64_t>* src_extra =
      src->extra.load(std::memory_order_relaxed);
  if (src_extra != nullptr) {
    EnsureExtra(dst);
    std::atomic<uint64_t>* dst_extra =
        dst->extra.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < n; ++i) {
      for (uint32_t t = 0; t < kExtraWords; ++t) {
        dst_extra[(dst_index + i) * kExtraWords + t].store(
            src_extra[(src_index + i) * kExtraWords + t].load(
                std::memory_order_relaxed),
            std::memory_order_relaxed);
      }
    }
  }
}

inline BSkipList::SlotView BSkipList::ReadSlot(const Node* node,
                                               uint32_t index) const {
  const Slot& src = node->slots[index];
  SlotView view;
  view.word0 = node->word0[index].load(std::memory_order_relaxed);
  view.word1 = src.word1.load(std::memory_order_relaxed);
  view.footer = src.footer.load(std::memory_order_relaxed);
  view.user_len = src.user_len.load(std::memory_order_relaxed);
  view.entry = src.entry.load(std::memory_order_relaxed);
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    view.extra[t] = 0;
  }
  const std::atomic<uint64_t>* extra =
      node->extra.load(std::memory_order_relaxed);
  if (extra != nullptr) {
    const std::atomic<uint64_t>* row = extra + index * kExtraWords;
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

inline int BSkipList::CompareKeyToSlotAt(const KeyView& key, const Node* node,
                                         uint32_t index) const {
  const Slot& slot = node->slots[index];
  const int c0 =
      CmpPacked(key.word0, node->word0[index].load(std::memory_order_relaxed));
  if (c0 != 0) {
    return c0;
  }
  // Only an occupied slot can match. Empty slots have word0 == 0, so a
  // non-zero key word still returns above without touching this line.
  if (slot.entry.load(std::memory_order_relaxed) == nullptr) {
    return 1;
  }
  const int c1 =
      CmpPacked(key.word1, slot.word1.load(std::memory_order_relaxed));
  if (c1 != 0) {
    return c1;
  }
  const uint32_t slot_len = slot.user_len.load(std::memory_order_relaxed);
  if (LIKELY(key.user_len <= kInlineUserKey && slot_len <= kInlineUserKey)) {
    if (key.user_len != slot_len) {
      return key.user_len < slot_len ? -1 : 1;
    }
    return CmpFooter(key.footer, slot.footer.load(std::memory_order_relaxed));
  }
  uint64_t extra[kExtraWords];
  const std::atomic<uint64_t>* row =
      node->extra.load(std::memory_order_relaxed);
  if (row != nullptr) {
    row += index * kExtraWords;
    for (uint32_t t = 0; t < kExtraWords; ++t) {
      extra[t] = row[t].load(std::memory_order_relaxed);
    }
  } else {
    for (uint32_t t = 0; t < kExtraWords; ++t) {
      extra[t] = 0;
    }
  }
  const char* other = nullptr;
  if (key.user_len > kPrefixBytes || slot_len > kPrefixBytes) {
    bool prefix_equal = true;
    for (uint32_t t = 0; t < kExtraWords; ++t) {
      if (key.extra[t] != extra[t]) {
        prefix_equal = false;
        break;
      }
    }
    if (prefix_equal) {
      const char* entry = slot.entry.load(std::memory_order_relaxed);
      other = GetLengthPrefixedSlice(entry).data();
    }
  }
  return CmpPastInline(key, extra, slot_len,
                       slot.footer.load(std::memory_order_relaxed), other);
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
  node->next_word0.store(next_node->word0[0].load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
  node->next_word1.store(slot0.word1.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
  const std::atomic<uint64_t>* extra =
      next_node->extra.load(std::memory_order_relaxed);
  for (uint32_t t = 0; t < kExtraWords; ++t) {
    uint64_t word = 0;
    if (extra != nullptr) {
      word = extra[t].load(std::memory_order_relaxed);
    }
    node->next_extra[t].store(word, std::memory_order_relaxed);
  }
}

inline BSkipList::Rank BSkipList::FindRank(const Node* node, uint32_t count,
                                           const KeyView& key) const {
  assert(count > 0);
  assert(count <= kFanout);
  const uint64_t kw0 = key.word0;
  const uint64_t kw1 = key.word1;
  uint32_t rank = 0;
  bool equal = false;
  for (uint32_t i = 1; i < count; ++i) {
    const uint64_t sw0 = node->word0[i].load(std::memory_order_relaxed);
    if (UNLIKELY(kw0 == sw0)) {
      const uint64_t sw1 = node->slots[i].word1.load(std::memory_order_relaxed);
      if (kw1 < sw1) {
        break;
      }
      if (kw1 > sw1) {
        rank = i;
        equal = false;
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
        const std::atomic<uint64_t>* row =
            node->extra.load(std::memory_order_relaxed);
        if (row != nullptr) {
          row += i * kExtraWords;
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
      equal = (cmp == 0);
      continue;
    }
    if (kw0 < sw0) {
      break;
    }
    rank = i;
    equal = false;
  }
  // Slot 0 is a real key on every node except a header sentinel, and the
  // search above starts at slot 1. The tree invariant puts `key` at or after
  // slot 0, so only an exact match needs the extra comparison.
  const Slot& head = node->slots[0];
  if (rank == 0 && head.entry.load(std::memory_order_relaxed) != nullptr &&
      node->word0[0].load(std::memory_order_relaxed) == kw0) {
    equal = CompareKeyToSlotAt(key, node, 0) == 0;
  }
  return {rank, equal};
}

inline BSkipList::Node* BSkipList::WalkRight(Node* node,
                                             const KeyView& key) const {
  for (;;) {
    // Acquire so the successor's constructor and slot stores, which happen
    // before the releasing store of this pointer, are visible. A relaxed load
    // lets a search observe a node that is still being built.
    Node* next = node->next.load(std::memory_order_acquire);
    if (next == nullptr) {
      return node;
    }
    // The next iteration needs the header of `next` (its next pointer and
    // cached next key). Start that miss while the comparison below runs.
    PREFETCH(next, 0, 3);
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
  // A release store orders earlier writes, not the data stores that follow.
  // Mark the lock with a relaxed store and then a release fence so readers
  // cannot observe those stores while the seqlock still looks even.
  node->seq.store(seq + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
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
  if (!node->seq.compare_exchange_strong(seq, seq + 1,
                                        std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
    return false;
  }
  // Same fence as BeginWrite: the CAS releases earlier work, not the writes
  // this critical section is about to publish.
  std::atomic_thread_fence(std::memory_order_release);
  return true;
}

inline void BSkipList::LockNode(Node* node) const {
  // Match SpinMutex. Pause-only spinning pins every waiter; if the owner is
  // descheduled, those waiters keep it off-CPU and the insert hangs.
  for (size_t tries = 0;; ++tries) {
    if (TryLockNode(node)) {
      return;
    }
    port::AsmVolatilePause();
    if (tries > 100) {
      std::this_thread::yield();
    }
  }
}

inline bool BSkipList::KeyBelongs(Node* node, const KeyView& key) const {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  if (count == 0 || count > kFanout) {
    return false;
  }
  if (CompareKeyToSlotAt(key, node, 0) < 0) {
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
  // A header exists at every level before height_ is published, and a
  // promotion links further nodes at that level first. Stopping at the
  // header skips those nodes, so every other inserter retries until the
  // publisher finishes. Walk the level anyway.
  int start = top > 0 ? top - 1 : 0;
  if (target_level > start) {
    start = target_level;
  }
  Node* node = headers_[start];
  for (int level = start; level >= target_level; --level) {
    node = WalkRight(node, key);
    if (node->level != static_cast<uint32_t>(level)) {
      return nullptr;
    }
    if (level == target_level) {
      return node;
    }
    const uint32_t seq1 = node->seq.load(std::memory_order_acquire);
    if ((seq1 & 1u) != 0) {
      return nullptr;
    }
    const uint32_t count = node->count.load(std::memory_order_relaxed);
    if (count == 0 || count > kFanout || node->children == nullptr) {
      return nullptr;
    }
    const Rank found = FindRank(node, count, key);
    Node* child =
        node->children[found.index].load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (node->seq.load(std::memory_order_relaxed) != seq1) {
      return nullptr;
    }
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
  // `prev` is only used by reverse iteration. Walk before taking the lock so
  // a long chain does not keep `right` odd, then store only if `pred` is
  // still the immediate predecessor. A splice that landed in between updates
  // `prev` itself.
  Node* pred = neu;
  for (;;) {
    Node* nxt = pred->next.load(std::memory_order_acquire);
    if (nxt == right || nxt == nullptr) {
      break;
    }
    pred = nxt;
  }
  LockNode(right);
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
                                                        Node* new_child,
                                                        Node* neu) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(rank < count);
  assert(neu != nullptr);
  WriteSlot(neu, 0, key);
  if (node->level > 0) {
    neu->children[0].store(new_child, std::memory_order_relaxed);
  }
  const uint32_t move = count - (rank + 1);
  CopySlots(neu, 1, node, rank + 1, move);
  neu->count.store(1 + move, std::memory_order_relaxed);
  SpliceLocked(node, neu, rank + 1);
  return neu;
}

inline bool BSkipList::FillSplitRight(Node* node, uint32_t pos,
                                      const KeyView& key, Node* new_child,
                                      Node* neu) const {
  // A full node plus one key is split in half. Copy the right half straight
  // out of the source node. The inserted key is either the first slot of
  // that half or a hole in the middle of it. The left node is not modified.
  assert(node->count.load(std::memory_order_relaxed) == kFanout);
  assert(pos > 0 && pos <= kFanout);
  assert(neu != nullptr);
  if (pos >= kSplitLeft) {
    const uint32_t head = pos - kSplitLeft;
    CopySlots(neu, 0, node, kSplitLeft, head);
    WriteSlot(neu, head, key);
    CopySlots(neu, head + 1, node, pos, kSplitRight - head - 1);
    if (neu->level > 0) {
      neu->children[head].store(new_child, std::memory_order_relaxed);
    }
  } else {
    CopySlots(neu, 0, node, kSplitLeft - 1, kSplitRight);
  }
  neu->count.store(kSplitRight, std::memory_order_relaxed);
  return pos >= kSplitLeft;
}

inline void BSkipList::PatchSplitLeft(Node* node, uint32_t pos,
                                      const KeyView& key, Node* new_child) {
  // The right sibling already holds the old slots from `kSplitLeft - 1` on.
  // Shift only the left half, so slot `kSplitLeft` is not rewritten.
  assert(pos > 0 && pos < kSplitLeft);
  if (pos + 1 < kSplitLeft) {
    ShiftSlotsRight(node, pos, kSplitLeft - 1);
  }
  WriteSlot(node, pos, key);
  if (node->level > 0) {
    node->children[pos].store(new_child, std::memory_order_relaxed);
  }
}

inline BSkipList::Node* BSkipList::SplitAndInsertLocked(Node* node,
                                                        uint32_t rank,
                                                        const KeyView& key,
                                                        Node* new_child,
                                                        Node* neu) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count == kFanout);
  assert(rank < count);
  const uint32_t pos = rank + 1;
  const bool on_right = FillSplitRight(node, pos, key, new_child, neu);
  if (!on_right) {
    PatchSplitLeft(node, pos, key, new_child);
  }
  Node* right = node->next.load(std::memory_order_relaxed);
  neu->prev.store(node, std::memory_order_relaxed);
  if (right != nullptr) {
    StoreNextHeader(neu, right);
  }
  neu->next.store(right, std::memory_order_release);
  node->count.store(kSplitLeft, std::memory_order_relaxed);
  StoreNextHeader(node, neu);
  node->next.store(neu, std::memory_order_release);
  EndWrite(node);
  FixRightPrev(neu, right);
  if (right == nullptr) {
    PublishTail(neu);
  }
  return on_right ? neu : node;
}

inline void BSkipList::ShiftSlotsRight(Node* node, uint32_t pos,
                                       uint32_t count) {
  // Move slots [pos, count) one position right. Copy each field stream
  // separately so the shift walks contiguous regions instead of alternating
  // between them for every slot.
  for (uint32_t i = count; i > pos; --i) {
    node->word0[i].store(node->word0[i - 1].load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
  }
  std::atomic<uint64_t>* extra = node->extra.load(std::memory_order_relaxed);
  if (extra != nullptr) {
    for (uint32_t i = count; i > pos; --i) {
      for (uint32_t t = 0; t < kExtraWords; ++t) {
        extra[i * kExtraWords + t].store(
            extra[(i - 1) * kExtraWords + t].load(std::memory_order_relaxed),
            std::memory_order_relaxed);
      }
    }
  }
  for (uint32_t i = count; i > pos; --i) {
    Slot& dst = node->slots[i];
    const Slot& src = node->slots[i - 1];
    dst.word1.store(src.word1.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
    dst.footer.store(src.footer.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
    dst.user_len.store(src.user_len.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
    dst.entry.store(src.entry.load(std::memory_order_relaxed),
                    std::memory_order_relaxed);
  }
  if (node->level > 0) {
    for (uint32_t i = count; i > pos; --i) {
      node->children[i].store(
          node->children[i - 1].load(std::memory_order_relaxed),
          std::memory_order_relaxed);
    }
  }
}

inline void BSkipList::ShiftInsertLocked(Node* node, uint32_t pos,
                                         const KeyView& key, Node* child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count < kFanout);
  assert(pos > 0);
  assert(pos <= count);
  ShiftSlotsRight(node, pos, count);
  WriteSlot(node, pos, key);
  if (node->level > 0) {
    node->children[pos].store(child, std::memory_order_relaxed);
  }
  node->count.store(count + 1, std::memory_order_relaxed);
  EndWrite(node);
}

inline bool BSkipList::TryAppendConcurrent(const KeyView& key, Node** landed) {
  // One spare node, reused across attempts. Allocating it while the tail
  // seqlock is held is what stalls every other appender.
  Node* spare = nullptr;
  for (int attempt = 0; attempt < 8; ++attempt) {
    Node* leaf = tail_.load(std::memory_order_acquire);
    if (leaf == nullptr || leaf->level != 0) {
      return false;
    }
    // Reject without locking. Random inserts must not queue on the tail.
    const uint32_t observed = leaf->count.load(std::memory_order_acquire);
    if (observed == 0 || observed > kFanout) {
      return false;
    }
    const bool append = CompareKeyToSlotAt(key, leaf, observed - 1) > 0;
    // Out-of-order keys that still belong in the tail node can be inserted
    // here too. Only keys from earlier nodes need the full search.
    if (!append && CompareKeyToSlotAt(key, leaf, 0) < 0) {
      return false;
    }
    if (leaf->next.load(std::memory_order_acquire) != nullptr &&
        CompareKeyToNextHeader(key, leaf) >= 0) {
      return false;
    }
    if (key.user_len > kInlineUserKey &&
        leaf->extra.load(std::memory_order_acquire) == nullptr) {
      EnsureExtra(leaf);
    }
    if (observed >= kFanout) {
      const bool copy_extra =
          key.user_len > kInlineUserKey ||
          leaf->extra.load(std::memory_order_acquire) != nullptr;
      if (spare == nullptr) {
        spare = NewNode(0);
      }
      if (copy_extra &&
          spare->extra.load(std::memory_order_acquire) == nullptr) {
        EnsureExtra(spare);
      }
    }
    if (!TryLockNode(leaf)) {
      port::AsmVolatilePause();
      continue;
    }
    const uint32_t count = leaf->count.load(std::memory_order_relaxed);
    if (count == 0 || count > kFanout || !KeyBelongs(leaf, key)) {
      EndWrite(leaf);
      return false;
    }
    if (append && CompareKeyToSlotAt(key, leaf, count - 1) > 0) {
      if (count < kFanout) {
        if (key.user_len > kInlineUserKey &&
            leaf->extra.load(std::memory_order_relaxed) == nullptr) {
          EndWrite(leaf);
          continue;
        }
        WriteSlot(leaf, count, key);
        leaf->count.store(count + 1, std::memory_order_relaxed);
        EndWrite(leaf);
        if (landed != nullptr) {
          *landed = leaf;
        }
        return true;
      }
      const bool copy_extra =
          key.user_len > kInlineUserKey ||
          leaf->extra.load(std::memory_order_relaxed) != nullptr;
      if (spare == nullptr ||
          (copy_extra &&
           spare->extra.load(std::memory_order_relaxed) == nullptr)) {
        EndWrite(leaf);
        continue;
      }
      constexpr uint32_t half = kFanout / 2;
      const uint32_t moved = kFanout - half;
      Node* neu = spare;
      spare = nullptr;
      CopySlots(neu, 0, leaf, half, moved);
      WriteSlot(neu, moved, key);
      neu->count.store(moved + 1, std::memory_order_relaxed);
      SpliceLocked(leaf, neu, half);
      if (landed != nullptr) {
        *landed = neu;
      }
      return true;
    }
    // The key belongs inside the tail node. Shift locally instead of walking
    // the tree, which is common when concurrent writers arrive out of order.
    if (key.user_len > kInlineUserKey &&
        leaf->extra.load(std::memory_order_relaxed) == nullptr) {
      EndWrite(leaf);
      continue;
    }
    const Rank found = FindRank(leaf, count, key);
    if (found.equal) {
      EndWrite(leaf);
      return false;
    }
    if (count < kFanout) {
      ShiftInsertLocked(leaf, found.index + 1, key, nullptr);
      if (landed != nullptr) {
        *landed = leaf;
      }
      return true;
    }
    const bool copy_extra =
        key.user_len > kInlineUserKey ||
        leaf->extra.load(std::memory_order_relaxed) != nullptr;
    if (spare == nullptr ||
        (copy_extra &&
         spare->extra.load(std::memory_order_relaxed) == nullptr)) {
      EndWrite(leaf);
      continue;
    }
    Node* dest = SplitAndInsertLocked(leaf, found.index, key, nullptr, spare);
    if (landed != nullptr) {
      *landed = dest;
    }
    return true;
  }
  return false;
}

inline BSkipList::Node* BSkipList::PromoteConcurrent(const KeyView& key,
                                                     int level,
                                                     Node* new_child) {
  Node* spare = nullptr;
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
    const Rank found = FindRank(node, count, key);
    if (found.equal) {
      EndWrite(node);
      return nullptr;
    }
    const bool copy_extra =
        key.user_len > kInlineUserKey ||
        node->extra.load(std::memory_order_relaxed) != nullptr;
    if (spare == nullptr ||
        (copy_extra &&
         spare->extra.load(std::memory_order_relaxed) == nullptr)) {
      EndWrite(node);
      if (spare == nullptr) {
        spare = NewNode(node->level);
      }
      if (copy_extra) {
        EnsureExtra(spare);
      }
      continue;
    }
    return PromotionSplitLocked(node, found.index, key, new_child, spare);
  }
}

inline bool BSkipList::InsertAtLevelConcurrent(const KeyView& key, int level,
                                               Node* child, Node** landed) {
  Node* spare = nullptr;
  for (;;) {
    Node* node = Locate(key, level);
    if (node == nullptr || node->level != static_cast<uint32_t>(level)) {
      port::AsmVolatilePause();
      continue;
    }
    if (key.user_len > kInlineUserKey &&
        node->extra.load(std::memory_order_acquire) == nullptr) {
      EnsureExtra(node);
    }
    LockNode(node);
    if (!KeyBelongs(node, key)) {
      EndWrite(node);
      continue;
    }
    const uint32_t count = node->count.load(std::memory_order_relaxed);
    const Rank found = FindRank(node, count, key);
    if (found.equal) {
      EndWrite(node);
      // A lower level of this insert already linked `key`. The separator is
      // present, so the key is in the list.
      return child != nullptr;
    }
    if (key.user_len > kInlineUserKey &&
        node->extra.load(std::memory_order_relaxed) == nullptr) {
      EndWrite(node);
      EnsureExtra(node);
      continue;
    }
    if (count >= kFanout) {
      const bool copy_extra =
          key.user_len > kInlineUserKey ||
          node->extra.load(std::memory_order_relaxed) != nullptr;
      if (spare == nullptr ||
          (copy_extra &&
           spare->extra.load(std::memory_order_relaxed) == nullptr)) {
        EndWrite(node);
        if (spare == nullptr) {
          spare = NewNode(static_cast<uint32_t>(level));
        }
        if (copy_extra) {
          EnsureExtra(spare);
        }
        continue;
      }
      Node* dest = SplitAndInsertLocked(node, found.index, key, child, spare);
      if (landed != nullptr) {
        *landed = dest;
      }
    } else {
      ShiftInsertLocked(node, found.index + 1, key, child);
      if (landed != nullptr) {
        *landed = node;
      }
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
  neu->next.store(right, std::memory_order_release);
  if (right != nullptr) {
    StoreNextHeader(neu, right);
  }
  BeginWrite(left);
  left->count.store(new_left_count, std::memory_order_relaxed);
  StoreNextHeader(left, neu);
  left->next.store(neu, std::memory_order_release);
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

inline BSkipList::InsertHint* BSkipList::PrepareHint(void** hint, bool on_heap) {
  if (*hint != nullptr) {
    return static_cast<InsertHint*>(*hint);
  }
  void* mem;
  if (on_heap) {
    mem = new char[sizeof(InsertHint)];
  } else {
    mem = allocator_->AllocateAligned(sizeof(InsertHint));
  }
  auto* created = new (mem) InsertHint();
  created->leaf = nullptr;
  *hint = mem;
  return created;
}

inline BSkipList::Node* BSkipList::FollowHint(Node* start,
                                              const KeyView& key) const {
  Node* node = start;
  // One extra leaf covers the split that moved the previous key. A longer
  // walk makes a hint miss scan the tail on every random insert.
  for (int step = 0; step < 2 && node != nullptr; ++step) {
    if (node->level != 0) {
      return nullptr;
    }
    const uint32_t count = node->count.load(std::memory_order_acquire);
    if (count == 0 || count > kFanout) {
      return nullptr;
    }
    if (CompareKeyToSlotAt(key, node, 0) < 0) {
      return nullptr;
    }
    Node* next = node->next.load(std::memory_order_acquire);
    if (next == nullptr || CompareKeyToNextHeader(key, node) < 0) {
      return node;
    }
    node = next;
  }
  return nullptr;
}

__attribute__((always_inline)) inline BSkipList::ApplyResult
BSkipList::AppendInLeaf(Node* leaf, const KeyView& key, Node** landed) {
  if (leaf == nullptr || leaf->level != 0) {
    return ApplyResult::kMiss;
  }
  const uint32_t count = leaf->count.load(std::memory_order_relaxed);
  // A full leaf takes the split path. That is 1 in 16 appends.
  if (count == 0 || count >= kFanout) {
    return ApplyResult::kMiss;
  }
  const int cmp_last = CompareKeyToSlotAt(key, leaf, count - 1);
  if (cmp_last == 0) {
    return ApplyResult::kDuplicate;
  }
  if (cmp_last < 0) {
    return ApplyResult::kMiss;
  }
  if (leaf->next.load(std::memory_order_relaxed) != nullptr &&
      CompareKeyToNextHeader(key, leaf) >= 0) {
    return ApplyResult::kMiss;
  }
  if (key.user_len > kInlineUserKey) {
    EnsureExtra(leaf);
  }
  BeginWrite(leaf);
  WriteSlot(leaf, count, key);
  leaf->count.store(count + 1, std::memory_order_relaxed);
  EndWrite(leaf);
  *landed = leaf;
  return ApplyResult::kInserted;
}

__attribute__((noinline)) inline BSkipList::ApplyResult
BSkipList::InsertIntoLeaf(Node* leaf,
                                                        const KeyView& key,
                                                        Node** landed) {
  if (leaf == nullptr || leaf->level != 0) {
    return ApplyResult::kMiss;
  }
  const uint32_t count = leaf->count.load(std::memory_order_relaxed);
  if (count == 0 || count > kFanout) {
    return ApplyResult::kMiss;
  }
  // One compare rejects a strict append's common case. Keys that sort inside
  // this leaf still insert here; only an earlier leaf needs a tree walk.
  const int cmp_last = CompareKeyToSlotAt(key, leaf, count - 1);
  if (cmp_last == 0) {
    return ApplyResult::kDuplicate;
  }
  if (leaf->next.load(std::memory_order_relaxed) != nullptr &&
      CompareKeyToNextHeader(key, leaf) >= 0) {
    return ApplyResult::kMiss;
  }
  if (cmp_last > 0) {
    if (count < kFanout) {
      if (key.user_len > kInlineUserKey) {
        EnsureExtra(leaf);
      }
      BeginWrite(leaf);
      WriteSlot(leaf, count, key);
      leaf->count.store(count + 1, std::memory_order_relaxed);
      EndWrite(leaf);
      *landed = leaf;
      return ApplyResult::kInserted;
    }
    constexpr uint32_t half = kFanout / 2;
    Node* neu = NewNode(0);
    const uint32_t moved = kFanout - half;
    if (key.user_len > kInlineUserKey ||
        leaf->extra.load(std::memory_order_acquire) != nullptr) {
      EnsureExtra(neu);
      if (key.user_len > kInlineUserKey) {
        EnsureExtra(leaf);
      }
    }
    CopySlots(neu, 0, leaf, half, moved);
    WriteSlot(neu, moved, key);
    neu->count.store(moved + 1, std::memory_order_relaxed);
    SpliceAfter(leaf, neu, half);
    *landed = neu;
    return ApplyResult::kInserted;
  }
  if (CompareKeyToSlotAt(key, leaf, 0) < 0) {
    return ApplyResult::kMiss;
  }
  const Rank found = FindRank(leaf, count, key);
  if (found.equal) {
    return ApplyResult::kDuplicate;
  }
  if (count < kFanout) {
    ShiftInsert(leaf, found.index + 1, key, nullptr);
    *landed = leaf;
    return ApplyResult::kInserted;
  }
  *landed = SplitAndInsert(leaf, found.index, key, nullptr);
  return ApplyResult::kInserted;
}

__attribute__((always_inline)) inline BSkipList::ApplyResult
BSkipList::TryAppend(const KeyView& key, Node** landed) {
  Node* leaf = tail_.load(std::memory_order_relaxed);
  const ApplyResult fast = AppendInLeaf(leaf, key, landed);
  if (fast != ApplyResult::kMiss) {
    return fast;
  }
  // The key is not a strict append. Insert it locally only when it still
  // belongs in the tail; a key from an earlier leaf must not stop here.
  if (leaf == nullptr || leaf->level != 0 ||
      CompareKeyToSlotAt(key, leaf, 0) < 0) {
    return ApplyResult::kMiss;
  }
  if (leaf->next.load(std::memory_order_relaxed) != nullptr &&
      CompareKeyToNextHeader(key, leaf) >= 0) {
    return ApplyResult::kMiss;
  }
  return InsertIntoLeaf(leaf, key, landed);
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
  CopySlots(neu, 1, node, rank + 1, move);
  neu->count.store(1 + move, std::memory_order_relaxed);
  SpliceAfter(node, neu, rank + 1);
  return neu;
}

inline BSkipList::Node* BSkipList::SplitAndInsert(Node* node, uint32_t rank,
                                                  const KeyView& key,
                                                  Node* new_child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count == kFanout);
  assert(rank < count);
  const uint32_t pos = rank + 1;
  Node* neu = NewNode(node->level);
  if (key.user_len > kInlineUserKey ||
      node->extra.load(std::memory_order_acquire) != nullptr) {
    EnsureExtra(neu);
    if (key.user_len > kInlineUserKey) {
      EnsureExtra(node);
    }
  }
  const bool on_right = FillSplitRight(node, pos, key, new_child, neu);
  Node* right = node->next.load(std::memory_order_relaxed);
  neu->prev.store(node, std::memory_order_relaxed);
  neu->next.store(right, std::memory_order_release);
  if (right != nullptr) {
    StoreNextHeader(neu, right);
  }
  BeginWrite(node);
  if (!on_right) {
    PatchSplitLeft(node, pos, key, new_child);
  }
  node->count.store(kSplitLeft, std::memory_order_relaxed);
  StoreNextHeader(node, neu);
  node->next.store(neu, std::memory_order_release);
  EndWrite(node);
  if (right != nullptr) {
    BeginWrite(right);
    right->prev.store(neu, std::memory_order_relaxed);
    EndWrite(right);
  }
  if (right == nullptr && node->level == 0) {
    tail_.store(neu, std::memory_order_release);
  }
  return on_right ? neu : node;
}

inline void BSkipList::ShiftInsert(Node* node, uint32_t pos, const KeyView& key,
                                   Node* child) {
  const uint32_t count = node->count.load(std::memory_order_relaxed);
  assert(count < kFanout);
  assert(pos > 0);
  assert(pos <= count);
  if (key.user_len > kInlineUserKey) {
    EnsureExtra(node);
  }
  BeginWrite(node);
  ShiftSlotsRight(node, pos, count);
  WriteSlot(node, pos, key);
  if (node->level > 0) {
    node->children[pos].store(child, std::memory_order_relaxed);
  }
  node->count.store(count + 1, std::memory_order_relaxed);
  EndWrite(node);
}

__attribute__((noinline)) inline bool BSkipList::InsertPrepared(const KeyView& key,
                                                            int height,
                                                            Node** remember,
                                                            bool try_tail) {
  if (try_tail && LIKELY(height == 0)) {
    Node* landed = nullptr;
    const ApplyResult appended = TryAppend(key, &landed);
    if (appended == ApplyResult::kInserted) {
      if (remember != nullptr) {
        *remember = landed;
      }
      num_entries_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    if (appended == ApplyResult::kDuplicate) {
      return false;
    }
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
    const uint32_t count = curr->count.load(std::memory_order_relaxed);
    const Rank found = FindRank(curr, count, key);
    if (found.equal) {
      if (remember != nullptr && level == 0) {
        *remember = curr;
      }
      return false;
    }
    path[level].node = curr;
    path[level].rank = found.index;
    if (level > 0) {
      curr = curr->children[found.index].load(std::memory_order_relaxed);
      assert(curr != nullptr);
    }
  }

  Node* landed = nullptr;
  Node* below = nullptr;
  for (int level = 0; level < height; ++level) {
    below = PromotionSplit(path[level].node, path[level].rank, key, below);
    if (level == 0) {
      landed = below;
    }
  }
  if (height == 0) {
    Node* node = path[0].node;
    const uint32_t rank = path[0].rank;
    const uint32_t count = node->count.load(std::memory_order_relaxed);
    if (count >= kFanout) {
      landed = SplitAndInsert(node, rank, key, nullptr);
    } else {
      ShiftInsert(node, rank + 1, key, nullptr);
      landed = node;
    }
  } else {
    Node* node = path[height].node;
    const uint32_t rank = path[height].rank;
    const uint32_t count = node->count.load(std::memory_order_relaxed);
    if (count >= kFanout) {
      SplitAndInsert(node, rank, key, below);
    } else {
      ShiftInsert(node, rank + 1, key, below);
    }
  }
  if (height + 1 > top) {
    height_.store(height + 1, std::memory_order_release);
  }
  num_entries_.fetch_add(1, std::memory_order_relaxed);
  if (remember != nullptr) {
    *remember = landed;
  }
  return true;
}

inline bool BSkipList::Insert(const char* entry) {
  const KeyView key = ParseEntry(entry);
  const int height = RandomHeight();
  // The not-full strict append is the bulk of a sequential load. Keep it in
  // this function so the search and split code stays off the hot path.
  if (LIKELY(height == 0)) {
    Node* landed = nullptr;
    const ApplyResult fast = TryAppend(key, &landed);
    if (fast == ApplyResult::kInserted) {
      num_entries_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    if (fast == ApplyResult::kDuplicate) {
      return false;
    }
  }
  return InsertPrepared(key, height, nullptr, false);
}

inline BSkipList::ApplyResult BSkipList::InsertIntoLeafConcurrent(
    Node* hint_leaf, const KeyView& key, Node** landed) {
  Node* spare = nullptr;
  Node* leaf = hint_leaf;
  for (int attempt = 0; attempt < 8; ++attempt) {
    leaf = FollowHint(leaf, key);
    if (leaf == nullptr) {
      return ApplyResult::kMiss;
    }
    const uint32_t observed = leaf->count.load(std::memory_order_acquire);
    if (observed == 0 || observed > kFanout) {
      return ApplyResult::kMiss;
    }
    const bool append = CompareKeyToSlotAt(key, leaf, observed - 1) > 0;
    if (key.user_len > kInlineUserKey &&
        leaf->extra.load(std::memory_order_acquire) == nullptr) {
      EnsureExtra(leaf);
    }
    if (observed >= kFanout) {
      const bool copy_extra =
          key.user_len > kInlineUserKey ||
          leaf->extra.load(std::memory_order_acquire) != nullptr;
      if (spare == nullptr) {
        spare = NewNode(0);
      }
      if (copy_extra &&
          spare->extra.load(std::memory_order_acquire) == nullptr) {
        EnsureExtra(spare);
      }
    }
    if (!TryLockNode(leaf)) {
      port::AsmVolatilePause();
      continue;
    }
    const uint32_t count = leaf->count.load(std::memory_order_relaxed);
    if (count == 0 || count > kFanout || !KeyBelongs(leaf, key)) {
      EndWrite(leaf);
      Node* next = leaf->next.load(std::memory_order_acquire);
      leaf = next == nullptr ? hint_leaf : next;
      continue;
    }
    if (append && CompareKeyToSlotAt(key, leaf, count - 1) > 0) {
      if (count < kFanout) {
        if (key.user_len > kInlineUserKey &&
            leaf->extra.load(std::memory_order_relaxed) == nullptr) {
          EndWrite(leaf);
          continue;
        }
        WriteSlot(leaf, count, key);
        leaf->count.store(count + 1, std::memory_order_relaxed);
        EndWrite(leaf);
        *landed = leaf;
        return ApplyResult::kInserted;
      }
      const bool copy_extra =
          key.user_len > kInlineUserKey ||
          leaf->extra.load(std::memory_order_relaxed) != nullptr;
      if (spare == nullptr ||
          (copy_extra &&
           spare->extra.load(std::memory_order_relaxed) == nullptr)) {
        EndWrite(leaf);
        continue;
      }
      constexpr uint32_t half = kFanout / 2;
      const uint32_t moved = kFanout - half;
      Node* neu = spare;
      spare = nullptr;
      CopySlots(neu, 0, leaf, half, moved);
      WriteSlot(neu, moved, key);
      neu->count.store(moved + 1, std::memory_order_relaxed);
      SpliceLocked(leaf, neu, half);
      *landed = neu;
      return ApplyResult::kInserted;
    }
    if (key.user_len > kInlineUserKey &&
        leaf->extra.load(std::memory_order_relaxed) == nullptr) {
      EndWrite(leaf);
      continue;
    }
    const Rank found = FindRank(leaf, count, key);
    if (found.equal) {
      EndWrite(leaf);
      return ApplyResult::kDuplicate;
    }
    if (count < kFanout) {
      ShiftInsertLocked(leaf, found.index + 1, key, nullptr);
      *landed = leaf;
      return ApplyResult::kInserted;
    }
    const bool copy_extra =
        key.user_len > kInlineUserKey ||
        leaf->extra.load(std::memory_order_relaxed) != nullptr;
    if (spare == nullptr ||
        (copy_extra &&
         spare->extra.load(std::memory_order_relaxed) == nullptr)) {
      EndWrite(leaf);
      continue;
    }
    Node* neu = spare;
    spare = nullptr;
    *landed = SplitAndInsertLocked(leaf, found.index, key, nullptr, neu);
    return ApplyResult::kInserted;
  }
  return ApplyResult::kMiss;
}

inline bool BSkipList::InsertConcurrentPrepared(const KeyView& key, int height,
                                                Node** remember) {
  if (LIKELY(height == 0)) {
    Node* landed = nullptr;
    if (TryAppendConcurrent(key, &landed)) {
      if (remember != nullptr) {
        *remember = landed;
      }
      num_entries_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }

  Node* landed = nullptr;
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
      if (remember != nullptr && landed != nullptr) {
        *remember = landed;
      }
      return true;
    }
    if (level == 0) {
      landed = created;
    }
    below = created;
  }
  if (!InsertAtLevelConcurrent(key, height, below,
                               height == 0 ? &landed : nullptr)) {
    return false;
  }
  PublishHeight(height);
  num_entries_.fetch_add(1, std::memory_order_relaxed);
  if (remember != nullptr) {
    *remember = landed;
  }
  return true;
}

inline bool BSkipList::InsertConcurrently(const char* entry) {
  return InsertConcurrentPrepared(ParseEntry(entry), RandomHeight(), nullptr);
}

inline bool BSkipList::InsertWithHint(const char* entry, void** hint) {
  assert(hint != nullptr);
  InsertHint* state = PrepareHint(hint, false);
  const KeyView key = ParseEntry(entry);
  const int height = RandomHeight();
  if (height == 0 && state->leaf != nullptr) {
    Node* landed = nullptr;
    const ApplyResult fast = AppendInLeaf(state->leaf, key, &landed);
    if (fast == ApplyResult::kInserted) {
      state->leaf = landed;
      num_entries_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    if (fast == ApplyResult::kDuplicate) {
      return false;
    }
    Node* target = FollowHint(state->leaf, key);
    if (target != nullptr) {
      const ApplyResult rc = InsertIntoLeaf(target, key, &landed);
      if (rc == ApplyResult::kInserted) {
        state->leaf = landed;
        num_entries_.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      if (rc == ApplyResult::kDuplicate) {
        return false;
      }
    }
  }
  Node* landed = nullptr;
  const bool ok = InsertPrepared(key, height, &landed);
  if (landed != nullptr) {
    state->leaf = landed;
  }
  return ok;
}

inline bool BSkipList::InsertWithHintConcurrently(const char* entry,
                                                  void** hint) {
  assert(hint != nullptr);
  InsertHint* state = PrepareHint(hint, true);
  const KeyView key = ParseEntry(entry);
  const int height = RandomHeight();
  if (height == 0 && state->leaf != nullptr) {
    Node* landed = nullptr;
    const ApplyResult rc =
        InsertIntoLeafConcurrent(state->leaf, key, &landed);
    if (rc == ApplyResult::kInserted) {
      state->leaf = landed;
      num_entries_.fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    if (rc == ApplyResult::kDuplicate) {
      return false;
    }
  }
  Node* landed = nullptr;
  const bool ok = InsertConcurrentPrepared(key, height, &landed);
  if (landed != nullptr) {
    state->leaf = landed;
  }
  return ok;
}

inline bool BSkipList::Descend(const KeyView& key, Node** leaf,
                               uint32_t* index, bool* equal) const {
  uint32_t blocked = 0;
  for (;;) {
    const int top = height_.load(std::memory_order_acquire);
    const int start = top > 0 ? top - 1 : 0;
    Node* node = headers_[start];
    bool failed = false;
    for (int level = start; level >= 0; --level) {
      for (;;) {
        const uint32_t seq1 = node->seq.load(std::memory_order_acquire);
        if (UNLIKELY((seq1 & 1u) != 0)) {
          // A reader that only pauses will keep a preempted writer off-CPU.
          port::AsmVolatilePause();
          if (++blocked > 100) {
            std::this_thread::yield();
          }
          failed = true;
          break;
        }
        blocked = 0;
        Node* next = node->next.load(std::memory_order_relaxed);
        if (next != nullptr) {
          PREFETCH(next, 0, 3);
        }
        const bool go_right =
            next != nullptr && CompareKeyToNextHeader(key, node) >= 0;
        uint32_t count = 0;
        uint32_t rank = 0;
        bool is_equal = false;
        Node* child = nullptr;
        if (!go_right) {
          count = node->count.load(std::memory_order_relaxed);
          if (count == 0 || count > kFanout) {
            failed = true;
            break;
          }
          const Rank found = FindRank(node, count, key);
          rank = found.index;
          is_equal = found.equal;
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
          *equal = is_equal;
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
