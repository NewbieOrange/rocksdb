// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under both the GPLv2 (found in the
// COPYING file in the root directory) and Apache 2.0 License
// (found in the LICENSE.Apache file in the root directory).

#include <atomic>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "db/dbformat.h"
#include "memory/arena.h"
#include "memory/concurrent_arena.h"
#include "memtable/bskiplist.h"
#include "rocksdb/comparator.h"
#include "port/port.h"
#include "rocksdb/convenience.h"
#include "rocksdb/db.h"
#include "rocksdb/memtablerep.h"
#include "test_util/testharness.h"
#include "util/coding.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {
namespace {

std::string Encode(const std::string& user, uint64_t seq) {
  std::string internal;
  internal.append(user);
  PutFixed64(&internal, PackSequenceAndType(seq, kTypeValue));
  std::string encoded;
  PutLengthPrefixedSlice(&encoded, internal);
  return encoded;
}

std::pair<std::string, uint64_t> Decode(const char* entry) {
  Slice internal = GetLengthPrefixedSlice(entry);
  assert(internal.size() >= 8);
  std::string user(internal.data(), internal.size() - 8);
  uint64_t seq = DecodeFixed64(internal.data() + internal.size() - 8) >> 8;
  return {user, seq};
}

struct Order {
  bool operator()(const std::pair<std::string, uint64_t>& a,
                  const std::pair<std::string, uint64_t>& b) const {
    if (a.first != b.first) {
      return a.first < b.first;
    }
    return a.second > b.second;
  }
};

using KeySet = std::set<std::pair<std::string, uint64_t>, Order>;

const char* Insert(BSkipList* list, const std::string& user, uint64_t seq) {
  std::string encoded = Encode(user, seq);
  char* buf = list->AllocateKey(encoded.size());
  memcpy(buf, encoded.data(), encoded.size());
  const bool inserted = list->Insert(buf);
  static_cast<void>(inserted);
  assert(inserted);
  return buf;
}

void ExpectMatches(const BSkipList& list, const KeySet& expected) {
  list.TEST_Validate();
  BSkipList::Iterator forward(&list);
  KeySet::const_iterator it = expected.begin();
  for (forward.SeekToFirst(); forward.Valid(); forward.Next(), ++it) {
    ASSERT_TRUE(it != expected.end());
    ASSERT_EQ(Decode(forward.key()), *it);
  }
  ASSERT_TRUE(it == expected.end());

  BSkipList::Iterator backward(&list);
  backward.SeekToLast();
  if (expected.empty()) {
    ASSERT_FALSE(backward.Valid());
    return;
  }
  it = expected.end();
  while (backward.Valid()) {
    ASSERT_TRUE(it != expected.begin());
    --it;
    ASSERT_EQ(Decode(backward.key()), *it);
    backward.Prev();
  }
  ASSERT_TRUE(it == expected.begin());
}

class BSkipListTest : public testing::Test {
 public:
  Arena arena_;
  BSkipList list_{&arena_};
};

TEST_F(BSkipListTest, Empty) {
  BSkipList::Iterator iter(&list_);
  iter.SeekToFirst();
  ASSERT_FALSE(iter.Valid());
  iter.SeekToLast();
  ASSERT_FALSE(iter.Valid());
  const char* encoded = Insert(&list_, "a", 1);
  static_cast<void>(encoded);
  // Seek on a fresh iterator before this insert is not the point. Re-check
  // empty behavior on a new list.
  Arena empty_arena;
  BSkipList empty(&empty_arena);
  const std::string target = Encode("a", 1);
  BSkipList::Iterator seeker(&empty);
  seeker.Seek(target.data());
  ASSERT_FALSE(seeker.Valid());
  seeker.SeekForPrev(target.data());
  ASSERT_FALSE(seeker.Valid());
  empty.TEST_Validate();
}

TEST_F(BSkipListTest, SequentialAndReverse) {
  KeySet expected;
  for (int i = 0; i < 1000; ++i) {
    char buf[16];
    snprintf(buf, sizeof(buf), "k%06d", i);
    ASSERT_TRUE(expected.insert({buf, 10}).second);
    Insert(&list_, buf, 10);
    if (i < 50 || i % 200 == 0) {
      ExpectMatches(list_, expected);
    }
  }
  ExpectMatches(list_, expected);
}

TEST_F(BSkipListTest, RandomOrder) {
  KeySet expected;
  Random rnd(301);
  for (int i = 0; i < 2000; ++i) {
    std::string user = "u" + std::to_string(rnd.Uniform(5000));
    uint64_t seq = rnd.Uniform(20) + 1;
    std::pair<std::string, uint64_t> item(user, seq);
    if (!expected.insert(item).second) {
      std::string encoded = Encode(user, seq);
      char* buf = list_.AllocateKey(encoded.size());
      memcpy(buf, encoded.data(), encoded.size());
      ASSERT_FALSE(list_.Insert(buf));
      continue;
    }
    Insert(&list_, user, seq);
  }
  ExpectMatches(list_, expected);

  BSkipList::Iterator iter(&list_);
  for (const auto& item : expected) {
    std::string encoded = Encode(item.first, item.second);
    iter.Seek(encoded.data());
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(Decode(iter.key()), item);
    ASSERT_TRUE(list_.Contains(encoded.data()));

    iter.SeekForPrev(encoded.data());
    ASSERT_TRUE(iter.Valid());
    ASSERT_EQ(Decode(iter.key()), item);
  }
}

TEST_F(BSkipListTest, SequenceOrderAndLongKeys) {
  KeySet expected;
  Insert(&list_, "same", 1);
  Insert(&list_, "same", 5);
  Insert(&list_, "same", 3);
  expected.insert({"same", 1});
  expected.insert({"same", 5});
  expected.insert({"same", 3});
  std::string encoded = Encode("same", 5);
  char* buf = list_.AllocateKey(encoded.size());
  memcpy(buf, encoded.data(), encoded.size());
  ASSERT_FALSE(list_.Insert(buf));

  const std::string long_key(64, 'a');
  const std::string long_key_b = std::string(64, 'a') + "b";
  Insert(&list_, long_key, 1);
  Insert(&list_, long_key_b, 1);
  Insert(&list_, long_key, 8);
  expected.insert({long_key, 1});
  expected.insert({long_key_b, 1});
  expected.insert({long_key, 8});
  ExpectMatches(list_, expected);
}

TEST_F(BSkipListTest, SeekBeforeAndAfter) {
  Insert(&list_, "b", 1);
  Insert(&list_, "d", 1);
  Insert(&list_, "f", 1);
  BSkipList::Iterator iter(&list_);
  const std::string before = Encode("a", 1);
  iter.Seek(before.data());
  ASSERT_EQ(Decode(iter.key()).first, "b");
  const std::string middle = Encode("c", 1);
  iter.Seek(middle.data());
  ASSERT_EQ(Decode(iter.key()).first, "d");
  const std::string after = Encode("z", 1);
  iter.Seek(after.data());
  ASSERT_FALSE(iter.Valid());
  iter.SeekForPrev(after.data());
  ASSERT_EQ(Decode(iter.key()).first, "f");
  iter.SeekForPrev(before.data());
  ASSERT_FALSE(iter.Valid());
}

TEST_F(BSkipListTest, ConcurrentReaders) {
  constexpr int kWritersKeys = 4000;
  std::atomic<bool> start{false};
  std::atomic<bool> failed{false};
  port::Thread writer([&]() {
    while (!start.load()) {
    }
    for (int i = 0; i < kWritersKeys; ++i) {
      char buf[16];
      snprintf(buf, sizeof(buf), "c%06d", i);
      Insert(&list_, buf, 1);
    }
  });
  std::vector<port::Thread> readers;
  for (int r = 0; r < 2; ++r) {
    readers.emplace_back([&]() {
      while (!start.load()) {
      }
      for (int n = 0; n < 50; ++n) {
        std::pair<std::string, uint64_t> prev;
        bool have_prev = false;
        BSkipList::Iterator iter(&list_);
        for (iter.SeekToFirst(); iter.Valid(); iter.Next()) {
          std::pair<std::string, uint64_t> cur = Decode(iter.key());
          if (have_prev && !(prev.first < cur.first ||
                             (prev.first == cur.first &&
                              prev.second > cur.second))) {
            failed.store(true);
            return;
          }
          prev = cur;
          have_prev = true;
        }
      }
    });
  }
  start.store(true);
  writer.join();
  for (port::Thread& reader : readers) {
    reader.join();
  }
  ASSERT_FALSE(failed.load());
  list_.TEST_Validate();
  BSkipList::Iterator iter(&list_);
  int count = 0;
  for (iter.SeekToFirst(); iter.Valid(); iter.Next()) {
    ++count;
  }
  ASSERT_EQ(count, kWritersKeys);
}

TEST(BSkipListMemtableTest, ConcurrentInserts) {
  const uint32_t seed = static_cast<uint32_t>(Env::Default()->NowMicros());
  SCOPED_TRACE("seed=" + std::to_string(seed));
  ConcurrentArena arena;
  BSkipList list(&arena);
  constexpr int kThreads = 4;
  constexpr int kPerThread = 400;
  std::mutex mu;
  KeySet expected;
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      Random rnd(seed + static_cast<uint32_t>(t) * 97u);
      KeySet local;
      auto add = [&](const std::string& user, uint64_t seq) {
        std::string encoded = Encode(user, seq);
        char* buf = list.AllocateKey(encoded.size());
        memcpy(buf, encoded.data(), encoded.size());
        if (list.InsertConcurrently(buf)) {
          local.insert({user, seq});
        }
      };
      for (int i = 0; i < kPerThread; ++i) {
        char user[16];
        snprintf(user, sizeof(user), "k%02d-%08d", t, i);
        add(user, 1);
        if (i % 32 == 0) {
          add("shared-" + std::to_string(i), 1);
        }
        if (i % 40 == 0) {
          const std::string long_key =
              std::string("long-key-") + user + "-padding";
          add(long_key, 3);
          add(long_key, 1);
        }
        char tail_key[16];
        snprintf(tail_key, sizeof(tail_key), "z%08d", t * kPerThread + i);
        add(tail_key, 1);
        if (rnd.OneIn(20)) {
          add("dup", static_cast<uint64_t>(t + 1));
        }
      }
      add("dup", 1);
      std::lock_guard<std::mutex> lock(mu);
      expected.insert(local.begin(), local.end());
    });
  }
  for (port::Thread& thread : threads) {
    thread.join();
  }
  ExpectMatches(list, expected);
}

TEST(BSkipListMemtableTest, ConcurrentInsertsAndDB) {
  const std::string dbname = test::PerThreadDBPath("bskiplist_db");
  ASSERT_OK(DestroyDB(dbname, Options()));
  Options options;
  options.create_if_missing = true;
  options.memtable_factory.reset(new BSkipListFactory());
  options.disable_auto_compactions = true;
  std::unique_ptr<DB> db;
  ASSERT_OK(DB::Open(options, dbname, &db));
  WriteOptions wopt;
  wopt.disableWAL = true;
  ASSERT_OK(db->Put(wopt, "b", "1"));
  ASSERT_OK(db->Put(wopt, "a", "1"));
  ASSERT_OK(db->Put(wopt, "a", "2"));
  std::string value;
  ASSERT_OK(db->Get(ReadOptions(), "a", &value));
  ASSERT_EQ(value, "2");
  ASSERT_OK(db->Get(ReadOptions(), "b", &value));
  ASSERT_EQ(value, "1");

  std::unique_ptr<Iterator> iter(db->NewIterator(ReadOptions()));
  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());
  ASSERT_EQ(iter->key().ToString(), "a");
  ASSERT_EQ(iter->value().ToString(), "2");
  iter->Next();
  ASSERT_EQ(iter->key().ToString(), "b");
  iter->Next();
  ASSERT_FALSE(iter->Valid());
  iter->SeekToLast();
  ASSERT_EQ(iter->key().ToString(), "b");
  iter->Prev();
  ASSERT_EQ(iter->key().ToString(), "a");
  iter->Prev();
  ASSERT_FALSE(iter->Valid());
  ASSERT_OK(iter->status());
  iter.reset();

  constexpr int kThreads = 4;
  constexpr int kPerThread = 200;
  std::vector<port::Thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([raw = db.get(), t]() {
      for (int i = 0; i < kPerThread; ++i) {
        std::string key = "t" + std::to_string(t) + "-" + std::to_string(i);
        ASSERT_OK(raw->Put(WriteOptions(), key, "v"));
      }
    });
  }
  for (port::Thread& thread : threads) {
    thread.join();
  }
  for (int t = 0; t < kThreads; ++t) {
    for (int i = 0; i < kPerThread; ++i) {
      std::string key = "t" + std::to_string(t) + "-" + std::to_string(i);
      ASSERT_OK(db->Get(ReadOptions(), key, &value));
      ASSERT_EQ(value, "v");
    }
  }
  db.reset();

  Options rejected = options;
  rejected.comparator = ReverseBytewiseComparator();
  rejected.create_if_missing = true;
  std::unique_ptr<DB> rejected_db;
  ASSERT_TRUE(DB::Open(rejected, dbname, &rejected_db).IsNotSupported());
  ASSERT_EQ(rejected_db, nullptr);
  ASSERT_OK(DestroyDB(dbname, options));
}

TEST(BSkipListMemtableTest, CreateFromString) {
  ConfigOptions config;
  std::unique_ptr<MemTableRepFactory> factory;
  ASSERT_OK(MemTableRepFactory::CreateFromString(config, "bskip_list",
                                                 &factory));
  ASSERT_NE(factory, nullptr);
  ASSERT_TRUE(factory->IsInstanceOf("bskip_list"));
  ASSERT_TRUE(factory->IsInstanceOf("BSkipListFactory"));
  ASSERT_TRUE(factory->IsInsertConcurrentlySupported());
  ASSERT_TRUE(factory->CanHandleDuplicatedKey());
}

}  // namespace
}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
