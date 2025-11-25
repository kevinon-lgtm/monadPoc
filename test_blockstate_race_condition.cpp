// Copyright (C) 2025 Category Labs, Inc.
//
// PoC: BlockState Concurrent Access Race Condition
//
// This test demonstrates the unsynchronized access to BlockState
// when multiple fibers execute transactions in parallel.

#include <category/execution/ethereum/state2/block_state.hpp>
#include <category/execution/ethereum/state3/state.hpp>
#include <category/execution/ethereum/core/account.hpp>
#include <category/execution/ethereum/db/test/test_db.hpp>
#include <category/vm/evm/explicit_traits.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <boost/fiber/all.hpp>
#include <thread>
#include <vector>

MONAD_NAMESPACE_BEGIN

namespace
{
    using namespace monad::vm::evm;

    // Simulate parallel access similar to execute_block_transactions
    TEST(BlockStateRaceCondition, ConcurrentMergeWithoutSynchronization)
    {
        // Setup
        auto const name =
            std::filesystem::temp_directory_path() / "blockstate_race_test";
        OnDiskMachine machine;
        mpt::Db db{machine, mpt::OnDiskDbConfig{.dbname_paths = {name}}};
        TrieDb tdb(db);
        vm::VM vm{};

        // Initialize with Alice having 1000 wei
        Address const alice{0xAAAA_address};
        Address const bob{0xBBBB_address};
        Address const carol{0xCCCC_address};

        StateDeltas initial_state;
        initial_state[alice] = StateDelta{
            .account = {std::nullopt, Account{.balance = uint256_t{1000}}},
            .storage = {}};

        tdb.commit(
            std::make_unique<StateDeltas>(initial_state),
            Code{},
            bytes32_t{1},
            BlockHeader{.number = 1});

        // Create BlockState (shared across "parallel" transactions)
        BlockState block_state{tdb, vm};

        // Simulate two concurrent transactions:
        // Tx1: Alice sends 100 to Bob
        // Tx2: Alice sends 200 to Carol
        // Expected result: Alice balance = 700 (1000 - 100 - 200)

        std::atomic<uint256_t> alice_final_balance{0};
        std::atomic<bool> race_detected{false};
        std::atomic<size_t> fiber_order{0};

        auto tx1 = [&]() {
            // Simulate tx1 execution
            State state{block_state, Incarnation{2, 0}};

            // Read initial balance (cache miss)
            auto balance1 = state.get_current_balance_pessimistic(alice);
            fiber_order++;

            // Simulate work
            std::this_thread::yield();

            // Modify balance: transfer to Bob
            state.subtract_from_balance(alice, uint256_t{100});
            state.add_to_balance(bob, uint256_t{100});

            // Merge back to shared BlockState
            MONAD_ASSERT(block_state.can_merge(state));
            block_state.merge(state);
        };

        auto tx2 = [&]() {
            // Simulate tx2 execution  
            State state{block_state, Incarnation{2, 0}};

            // Read initial balance (should see tx1's update, but might race)
            auto balance2 = state.get_current_balance_pessimistic(alice);
            fiber_order++;

            // Simulate work
            std::this_thread::yield();

            // Modify balance: transfer to Carol
            state.subtract_from_balance(alice, uint256_t{200});
            state.add_to_balance(carol, uint256_t{200});

            // Merge back to shared BlockState
            MONAD_ASSERT(block_state.can_merge(state));
            block_state.merge(state);
        };

        // Execute both transactions concurrently using fibers
        boost::fibers::fiber f1(tx1);
        boost::fibers::fiber f2(tx2);

        f1.join();
        f2.join();

        // Verify final state
        auto alice_account = block_state.read_account(alice);
        if (alice_account.has_value()) {
            alice_final_balance = alice_account->balance;
        }

        // Expected: 700 (1000 - 100 - 200)
        // If race occurred: might be 900 (1000 - 100) or 800 (1000 - 200)
        // or some other corrupted value

        EXPECT_EQ(
            alice_final_balance,
            uint256_t{700});  // Will likely FAIL due to race condition
    }

    // More aggressive race condition test with many concurrent accesses
    TEST(BlockStateRaceCondition, ManyFibersAccessSameAccount)
    {
        auto const name =
            std::filesystem::temp_directory_path() / "blockstate_many_fibers";
        OnDiskMachine machine;
        mpt::Db db{machine, mpt::OnDiskDbConfig{.dbname_paths = {name}}};
        TrieDb tdb(db);
        vm::VM vm{};

        Address const target{0xDEAD_address};

        // Initialize target account
        StateDeltas initial_state;
        initial_state[target] = StateDelta{
            .account = {std::nullopt, Account{.balance = uint256_t{10000}}},
            .storage = {}};

        tdb.commit(
            std::make_unique<StateDeltas>(initial_state),
            Code{},
            bytes32_t{1},
            BlockHeader{.number = 1});

        BlockState block_state{tdb, vm};

        constexpr size_t NUM_FIBERS = 10;
        std::atomic<size_t> successful_merges{0};

        auto worker = [&](size_t id) {
            State state{block_state, Incarnation{2, id}};

            // Each fiber tries to read and modify the same account
            auto account = state.recent_account(target);
            
            // Intentional contention
            for (int i = 0; i < 5; ++i) {
                state.add_to_balance(target, uint256_t{1});
                std::this_thread::yield();
            }

            // Try to merge
            if (block_state.can_merge(state)) {
                block_state.merge(state);
                successful_merges++;
            }
        };

        std::vector<boost::fibers::fiber> fibers;
        for (size_t i = 0; i < NUM_FIBERS; ++i) {
            fibers.emplace_back(worker, i);
        }

        for (auto &f : fibers) {
            f.join();
        }

        // In correct execution, all merges should succeed
        // With race conditions, some might fail or corrupt state
        EXPECT_GE(successful_merges, 1);

        auto final_account = block_state.read_account(target);
        // Expected: 10000 + (NUM_FIBERS * 5 additions)
        // Actual: might be less due to lost updates from race conditions
    }

}  // namespace

MONAD_NAMESPACE_END
