#include "common-checkpoint-db.h"
#include "common.h"
#include "log.h"
#include "testing.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string tmpdir() {
    char tmpl[] = "/tmp/ckptdb-test-XXXXXX";
    char * dir = mkdtemp(tmpl);
    if (!dir) {
        perror("mkdtemp");
        exit(1);
    }
    return std::string(dir);
}

static llama_tokens make_tokens(const std::vector<int> & ids) {
    llama_tokens t;
    t.reserve(ids.size());
    for (int id : ids) {
        t.push_back((llama_token)id);
    }
    return t;
}

static std::vector<uint8_t> make_kv(size_t sz) {
    std::vector<uint8_t> v(sz);
    for (size_t i = 0; i < sz; ++i) {
        v[i] = (uint8_t)(i ^ 0xAB);
    }
    return v;
}

int main() {
    testing t;

    // 1. Basic store + exact-match find
    t.test("basic store + find", [&](testing &t) {
        auto dir = tmpdir();
        checkpoint_db_config cfg;
        cfg.disk_path        = dir;
        cfg.ram_capacity_mib = 0;
        cfg.disk_capacity_mib = 1024;

        checkpoint_db db(cfg);

        auto tokens = make_tokens({100, 200, 300, 400});
        auto kv     = make_kv(64);
        std::list<common_prompt_checkpoint> cps;
        common_prompt_checkpoint cp;
        cp.n_tokens = 2; cp.pos_min = 0; cp.pos_max = 1;
        cp.data_tgt = make_kv(16);
        cps.push_back(std::move(cp));

        db.store(tokens, kv, {}, cps);

        auto match = db.find(tokens);
        t.assert_true("exact match found", match.found);
        t.assert_true("n_matched == 4", match.n_matched == 4);
        t.assert_true("KV size 64", match.kv_main.size() == 64);
        t.assert_true("no draft KV", match.kv_drft.empty());
        t.assert_true("1 checkpoint", match.checkpoints.size() == 1);
        t.assert_true("tokens size 4", match.matched_tokens.size() == 4);
        for (size_t i = 0; i < 64; ++i) {
            if (match.kv_main[i] != (uint8_t)(i ^ 0xAB)) {
                t.assert_true("KV integrity at " + std::to_string(i), false);
                break;
            }
        }
        fs::remove_all(dir);
    });

    // 2. No match
    t.test("no match", [&](testing &t) {
        auto dir = tmpdir();
        checkpoint_db_config cfg;
        cfg.disk_path = dir; cfg.ram_capacity_mib = 0; cfg.disk_capacity_mib = 1024;
        checkpoint_db db(cfg);
        db.store(make_tokens({100, 200, 300}), make_kv(16), {}, {});
        auto match = db.find(make_tokens({999, 888}));
        t.assert_true("no match for different tokens", !match.found);
        fs::remove_all(dir);
    });

    // 3. Empty query
    t.test("empty query", [&](testing &t) {
        checkpoint_db_config cfg;
        cfg.ram_capacity_mib = 64; cfg.disk_path = "";
        checkpoint_db db(cfg);
        db.store(make_tokens({1, 2, 3}), make_kv(8), {}, {});
        auto match = db.find({});
        t.assert_true("empty query no match", !match.found);

        checkpoint_db db2(cfg);
        auto match2 = db2.find(make_tokens({1, 2, 3}));
        t.assert_true("empty DB no match", !match2.found);
    });

    // 4. Persistence (flush + load_manifest)
    t.test("flush + load persistence", [&](testing &t) {
        auto dir = tmpdir();
        {
            checkpoint_db_config cfg;
            cfg.disk_path = dir; cfg.ram_capacity_mib = 0; cfg.disk_capacity_mib = 1024;
            checkpoint_db db(cfg);
            db.store(make_tokens({10, 20, 30, 40}), make_kv(128), make_kv(64), {});
            db.store(make_tokens({10, 20, 99}),      make_kv(32),  {},          {});
            db.flush();
        }
        {
            checkpoint_db_config cfg;
            cfg.disk_path = dir; cfg.ram_capacity_mib = 0; cfg.disk_capacity_mib = 1024;
            checkpoint_db db(cfg);
            db.load_manifest();
            t.assert_true("2 entries after reload", db.n_entries() == 2);

            auto m1 = db.find(make_tokens({10, 20, 30, 40}));
            t.assert_true("entry 1 found", m1.found);
            t.assert_true("entry 1 matched 4", m1.n_matched == 4);
            t.assert_equal((size_t)128, m1.kv_main.size());
            t.assert_equal((size_t)64,  m1.kv_drft.size());

            auto m2 = db.find(make_tokens({10, 20, 99}));
            t.assert_true("entry 2 found", m2.found);
            t.assert_true("entry 2 matched 3", m2.n_matched == 3);

            auto m3 = db.find(make_tokens({10, 20, 30, 99}));
            t.assert_true("LCP match after reload", m3.found);
            t.assert_true("LCP = 3", m3.n_matched == 3);
        }
        fs::remove_all(dir);
    });

    // 5. RAM eviction with disk spill
    t.test("RAM eviction", [&](testing &t) {
        auto dir = tmpdir();
        checkpoint_db_config cfg;
        cfg.disk_path = dir; cfg.ram_capacity_mib = 0; cfg.disk_capacity_mib = 1;
        checkpoint_db db(cfg);
        db.store(make_tokens({1}), make_kv(100), {}, {});
        db.store(make_tokens({2}), make_kv(200 * 1024), {}, {});
        auto match = db.find(make_tokens({1}));
        t.assert_true("entry 1 still findable from disk", match.found);
        t.assert_true("KV size 100", match.kv_main.size() == 100);
        fs::remove_all(dir);
    });

    // 6. Draft KV round-trip
    t.test("draft KV round-trip", [&](testing &t) {
        checkpoint_db_config cfg;
        cfg.ram_capacity_mib = 64; cfg.disk_path = "";
        checkpoint_db db(cfg);
        db.store(make_tokens({42, 43, 44}), make_kv(256), make_kv(128), {});
        auto match = db.find(make_tokens({42, 43, 44}));
        t.assert_true("match found", match.found);
        t.assert_equal((size_t)256, match.kv_main.size());
        t.assert_equal((size_t)128, match.kv_drft.size());
        for (size_t i = 0; i < 256; ++i) {
            if (match.kv_main[i] != (uint8_t)(i ^ 0xAB)) {
                t.assert_true("main KV integrity at " + std::to_string(i), false);
                break;
            }
        }
    });

    // 7. Checkpoints round-trip through persistence
    t.test("checkpoints persistence", [&](testing &t) {
        auto dir = tmpdir();
        checkpoint_db_config cfg;
        cfg.disk_path = dir; cfg.ram_capacity_mib = 0; cfg.disk_capacity_mib = 1024;
        checkpoint_db db(cfg);

        std::list<common_prompt_checkpoint> cps;
        for (int i = 0; i < 3; ++i) {
            common_prompt_checkpoint cp;
            cp.n_tokens = 10 + i; cp.pos_min = i*5; cp.pos_max = i*5+4;
            cp.data_tgt = make_kv(32 + i*8); cp.data_dft = make_kv(16 + i*4);
            cps.push_back(std::move(cp));
        }
        db.store(make_tokens({1,2,3,4,5}), make_kv(64), make_kv(32), cps);
        db.flush();

        checkpoint_db db2(cfg);
        db2.load_manifest();
        auto match = db2.find(make_tokens({1,2,3,4,5}));
        t.assert_true("match after reload", match.found);
        t.assert_equal((size_t)3, match.checkpoints.size());
        int i = 0;
        for (const auto & cp : match.checkpoints) {
            t.assert_equal((int64_t)(10 + i), cp.n_tokens);
            t.assert_equal(i*5, (int)cp.pos_min);
            t.assert_equal(i*5+4, (int)cp.pos_max);
            t.assert_equal((size_t)(32 + i*8), cp.data_tgt.size());
            t.assert_equal((size_t)(16 + i*4), cp.data_dft.size());
            ++i;
        }
        fs::remove_all(dir);
    });

    // 8. Best LCP across many entries
    t.test("best LCP across many entries", [&](testing &t) {
        checkpoint_db_config cfg;
        cfg.ram_capacity_mib = 64; cfg.disk_path = "";
        checkpoint_db db(cfg);
        db.store(make_tokens({1,2,3,4,5,6}), make_kv(8), {}, {});
        db.store(make_tokens({1,2,3,7,8}),   make_kv(8), {}, {});
        db.store(make_tokens({1,9,10}),       make_kv(8), {}, {});
        db.store(make_tokens({99,98}),        make_kv(8), {}, {});

        auto m1 = db.find(make_tokens({1,2,3,99,99}));
        t.assert_true("LCP 3 found", m1.found && m1.n_matched == 3);

        auto m2 = db.find(make_tokens({1,2,3,4,99}));
        t.assert_true("LCP 4 found", m2.found && m2.n_matched == 4);

        auto m3 = db.find(make_tokens({1,99,99}));
        t.assert_true("LCP 1 found", m3.found && m3.n_matched == 1);

        auto m4 = db.find(make_tokens({100,101}));
        t.assert_true("no match", !m4.found);
    });

    // 9. RAM-only mode (no disk path)
    t.test("RAM-only mode", [&](testing &t) {
        checkpoint_db_config cfg;
        cfg.ram_capacity_mib = 64; cfg.disk_path = "";
        checkpoint_db db(cfg);
        db.store(make_tokens({1,2,3}), make_kv(16), {}, {});
        db.store(make_tokens({4,5,6}), make_kv(16), {}, {});
        auto match = db.find(make_tokens({1,2,3}));
        t.assert_true("match in RAM-only", match.found);
        db.flush(); // no-op, should not crash
    });

    // 10. Overwrite same prefix
    t.test("overwrite same prefix", [&](testing &t) {
        checkpoint_db_config cfg;
        cfg.ram_capacity_mib = 64; cfg.disk_path = "";
        checkpoint_db db(cfg);
        db.store(make_tokens({1,2,3}), make_kv(16), {}, {});
        db.store(make_tokens({1,2,3}), make_kv(32), {}, {});
        auto match = db.find(make_tokens({1,2,3}));
        t.assert_true("match found", match.found);
        t.assert_equal((size_t)32, match.kv_main.size());
    });

    // 11. Many entries stress test
    t.test("many entries stress test", [&](testing &t) {
        auto dir = tmpdir();
        checkpoint_db_config cfg;
        cfg.disk_path = dir; cfg.ram_capacity_mib = 0; cfg.disk_capacity_mib = 512;
        checkpoint_db db(cfg);
        const int N = 500;
        for (int i = 0; i < N; ++i) {
            db.store(make_tokens({1, 2, i}), make_kv(64), {}, {});
        }
        t.assert_equal(N, (int)db.n_entries());
        for (int i = 0; i < N; ++i) {
            auto match = db.find(make_tokens({1, 2, i}));
            t.assert_true("exact match " + std::to_string(i), match.found && match.n_matched == 3);
        }
        auto lcp = db.find(make_tokens({1, 2, 99999}));
        t.assert_true("LCP match", lcp.found && lcp.n_matched == 2);

        db.flush();
        checkpoint_db db2(cfg);
        db2.load_manifest();
        t.assert_equal(N, (int)db2.n_entries());
        for (int i = 0; i < N; i += 50) {
            auto m = db2.find(make_tokens({1, 2, i}));
            t.assert_true("post-reload match " + std::to_string(i), m.found);
        }
        fs::remove_all(dir);
    });

    // 12. Large KV blob
    t.test("large KV blob", [&](testing &t) {
        auto dir = tmpdir();
        checkpoint_db_config cfg;
        cfg.disk_path = dir; cfg.ram_capacity_mib = 0; cfg.disk_capacity_mib = 1024;
        checkpoint_db db(cfg);
        const size_t KV_SIZE = 4 * 1024 * 1024;
        db.store(make_tokens({1,2,3,4,5}), make_kv(KV_SIZE), {}, {});
        db.flush();

        checkpoint_db db2(cfg);
        db2.load_manifest();
        auto match = db2.find(make_tokens({1,2,3,4,5}));
        t.assert_true("large KV match", match.found);
        t.assert_equal(KV_SIZE, match.kv_main.size());
        t.assert_true("byte 0", match.kv_main[0] == (uint8_t)(0 ^ 0xAB));
        t.assert_true("byte 999999", match.kv_main[999999] == (uint8_t)(999999 ^ 0xAB));
        t.assert_true("last byte", match.kv_main[KV_SIZE-1] == (uint8_t)((KV_SIZE-1) ^ 0xAB));
        fs::remove_all(dir);
    });

    // 13. Empty store + find
    t.test("empty KV store", [&](testing &t) {
        checkpoint_db_config cfg;
        cfg.ram_capacity_mib = 64; cfg.disk_path = "";
        checkpoint_db db(cfg);
        db.store(make_tokens({1,2,3}), {}, {}, {});
        auto match = db.find(make_tokens({1,2,3}));
        t.assert_true("match with empty KV", match.found);
        t.assert_true("empty KV main", match.kv_main.empty());
        t.assert_true("empty KV drft", match.kv_drft.empty());
        t.assert_true("no checkpoints", match.checkpoints.empty());
    });

    // 14. Single token
    t.test("single token", [&](testing &t) {
        checkpoint_db_config cfg;
        cfg.ram_capacity_mib = 64; cfg.disk_path = "";
        checkpoint_db db(cfg);
        db.store(make_tokens({42}), make_kv(8), {}, {});
        auto match = db.find(make_tokens({42}));
        t.assert_true("match", match.found && match.n_matched == 1);
        auto no_match = db.find(make_tokens({99}));
        t.assert_true("no match", !no_match.found);
    });

    return t.summary();
}
