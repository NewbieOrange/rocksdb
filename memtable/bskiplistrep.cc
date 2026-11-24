// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#include "db/memtable.h"
#include "memtable/bskiplist.h"
#include "rocksdb/comparator.h"
#include "rocksdb/memtablerep.h"

namespace ROCKSDB_NAMESPACE {
namespace {

class BSkipListRep : public MemTableRep {
 public:
  explicit BSkipListRep(Allocator* allocator)
      : MemTableRep(allocator), list_(allocator) {}

  KeyHandle Allocate(const size_t len, char** buf) override {
    *buf = list_.AllocateKey(len);
    return static_cast<KeyHandle>(*buf);
  }

  void Insert(KeyHandle handle) override {
    const bool inserted = InsertKey(handle);
    static_cast<void>(inserted);
    assert(inserted);
  }

  bool InsertKey(KeyHandle handle) override {
    return list_.Insert(static_cast<char*>(handle));
  }

  bool InsertKeyWithHint(KeyHandle handle, void** hint) override {
    return list_.InsertWithHint(static_cast<char*>(handle), hint);
  }

  void InsertConcurrently(KeyHandle handle) override {
    const bool inserted = InsertKeyConcurrently(handle);
    static_cast<void>(inserted);
    assert(inserted);
  }

  bool InsertKeyConcurrently(KeyHandle handle) override {
    return list_.InsertConcurrently(static_cast<char*>(handle));
  }

  bool InsertKeyWithHintConcurrently(KeyHandle handle, void** hint) override {
    return list_.InsertWithHintConcurrently(static_cast<char*>(handle), hint);
  }

  bool Contains(const char* key) const override { return list_.Contains(key); }

  size_t ApproximateMemoryUsage() override { return 0; }

  void Get(const LookupKey& key, void* callback_args,
           bool (*callback_func)(void* arg, const char* entry)) override {
    BSkipList::Iterator iter(&list_);
    for (iter.Seek(key.memtable_key().data());
         iter.Valid() && callback_func(callback_args, iter.key());
         iter.Next()) {
    }
  }

  uint64_t ApproximateNumEntries(const Slice& start_ikey,
                                 const Slice& end_ikey) override {
    return list_.ApproximateNumEntries(start_ikey, end_ikey);
  }

  void UniqueRandomSample(const uint64_t /*num_entries*/,
                          const uint64_t target_sample_size,
                          std::unordered_set<const char*>* entries) override {
    list_.UniqueRandomSample(target_sample_size, entries);
  }

  class Iterator : public MemTableRep::Iterator {
   public:
    explicit Iterator(const BSkipList* list) : iter_(list) {}

    bool Valid() const override { return iter_.Valid(); }

    const char* key() const override { return iter_.key(); }

    void Next() override { iter_.Next(); }

    void Prev() override { iter_.Prev(); }

    void Seek(const Slice& internal_key, const char* memtable_key) override {
      if (memtable_key != nullptr) {
        iter_.Seek(memtable_key);
      } else {
        iter_.Seek(EncodeKey(&tmp_, internal_key));
      }
    }

    void SeekForPrev(const Slice& internal_key,
                     const char* memtable_key) override {
      if (memtable_key != nullptr) {
        iter_.SeekForPrev(memtable_key);
      } else {
        iter_.SeekForPrev(EncodeKey(&tmp_, internal_key));
      }
    }

    void SeekToFirst() override { iter_.SeekToFirst(); }

    void SeekToLast() override { iter_.SeekToLast(); }

   private:
    BSkipList::Iterator iter_;
    std::string tmp_;
  };

  MemTableRep::Iterator* GetIterator(Arena* arena = nullptr) override {
    if (arena == nullptr) {
      return new Iterator(&list_);
    }
    void* mem = arena->AllocateAligned(sizeof(Iterator));
    return new (mem) Iterator(&list_);
  }

 private:
  BSkipList list_;
};

}  // namespace

MemTableRep* BSkipListFactory::CreateMemTableRep(
    const MemTableRep::KeyComparator& /*compare*/, Allocator* allocator,
    const SliceTransform* /*transform*/, Logger* /*logger*/) {
  return new BSkipListRep(allocator);
}

Status BSkipListFactory::ValidateOptions(
    const DBOptions& db_opts, const ColumnFamilyOptions& cf_opts) const {
  const Comparator* const ucmp = cf_opts.comparator;
  assert(ucmp != nullptr);
  if (ucmp != BytewiseComparator()) {
    return Status::NotSupported(
        "BSkipListFactory requires the default bytewise comparator");
  }
  return MemTableRepFactory::ValidateOptions(db_opts, cf_opts);
}

}  // namespace ROCKSDB_NAMESPACE
