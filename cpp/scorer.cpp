/*
 * GPU-Accelerated Portfolio Optimization Engine
 * C++/OpenMP Asset Combination Scorer
 *
 * Parallelizes gradient-boosted decision tree inference for scoring
 * 10K+ asset combinations per second.
 *
 * Loads trained model from data/scorer_model.json (exported by train_scorer.py),
 * falling back to a built-in demo model if the JSON file is not found.
 *
 * Compile:
 *   g++ -O3 -fopenmp -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       -o scorer$(python3-config --extension-suffix) scorer.cpp
 *
 * Or use CMake (see CMakeLists.txt)
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>

#ifdef USE_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

// ==============================================================================
// Decision Tree Node for asset scoring
// ==============================================================================
struct TreeNode {
    int feature_idx;    // which feature to split on
    float threshold;    // split threshold
    float value;        // leaf value (prediction)
    int left;           // left child index (-1 if leaf)
    int right;          // right child index (-1 if leaf)
};

// ==============================================================================
// Simple Gradient-Boosted Tree Ensemble
// ==============================================================================
struct TreeEnsemble {
    std::vector<std::vector<TreeNode>> trees;
    float learning_rate;
    float base_prediction;

    float predict(const std::vector<float>& features) const {
        float pred = base_prediction;
        for (const auto& tree : trees) {
            int node = 0;
            while (tree[node].left != -1) {
                if (features[tree[node].feature_idx] <= tree[node].threshold)
                    node = tree[node].left;
                else
                    node = tree[node].right;
            }
            pred += learning_rate * tree[node].value;
        }
        return pred;
    }
};

// ==============================================================================
// Minimal JSON token reader (no external dependencies)
// Parses the scorer_model.json format exported by train_scorer.py.
// ==============================================================================

// Skip whitespace in a string starting at pos
static void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos]==' '||s[pos]=='\n'||s[pos]=='\r'||s[pos]=='\t')) ++pos;
}

// Read a JSON number (int or float) as double, advance pos past it
static double read_number(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    while (pos < s.size() && (std::isdigit(s[pos]) || s[pos]=='.' || s[pos]=='e' || s[pos]=='E' || s[pos]=='+' || s[pos]=='-')) {
        if (pos > start && (s[pos]=='+' || s[pos]=='-') && s[pos-1]!='e' && s[pos-1]!='E') break;
        ++pos;
    }
    return std::stod(s.substr(start, pos - start));
}

// Skip to past the next occurrence of character c
static void skip_to(const std::string& s, size_t& pos, char c) {
    while (pos < s.size() && s[pos] != c) ++pos;
    if (pos < s.size()) ++pos;
}

// Read a JSON string (positioned at opening quote), return contents
static std::string read_string(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '"') ++pos; // skip opening "
    size_t start = pos;
    while (pos < s.size() && s[pos] != '"') ++pos;
    std::string result = s.substr(start, pos - start);
    if (pos < s.size()) ++pos; // skip closing "
    return result;
}

// Load a TreeEnsemble from scorer_model.json
// Returns true on success, false if file not found or parse error.
bool load_model_from_json(const std::string& path, TreeEnsemble& model) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::ostringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();
    file.close();

    // Parse top-level fields
    size_t pos = 0;

    // Find learning_rate
    auto find_key = [&](const std::string& key) -> bool {
        size_t found = json.find("\"" + key + "\"", pos);
        if (found == std::string::npos) { pos = 0; found = json.find("\"" + key + "\""); }
        if (found == std::string::npos) return false;
        pos = found + key.size() + 2; // past the closing quote
        skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ':') ++pos;
        skip_ws(json, pos);
        return true;
    };

    if (find_key("learning_rate"))  model.learning_rate  = (float)read_number(json, pos);
    if (find_key("base_prediction")) model.base_prediction = (float)read_number(json, pos);

    // Parse trees array
    size_t trees_start = json.find("\"trees\"");
    if (trees_start == std::string::npos) return false;
    pos = trees_start;
    skip_to(json, pos, '['); // outer array of trees

    model.trees.clear();

    // Each tree is an array of node objects
    while (pos < json.size()) {
        skip_ws(json, pos);
        if (pos >= json.size() || json[pos] == ']') break; // end of trees array
        if (json[pos] == ',') { ++pos; continue; }
        if (json[pos] != '[') break; // expect start of a tree array
        ++pos; // skip [

        std::vector<TreeNode> tree;
        while (pos < json.size()) {
            skip_ws(json, pos);
            if (pos >= json.size() || json[pos] == ']') { ++pos; break; }
            if (json[pos] == ',') { ++pos; continue; }
            if (json[pos] != '{') break;
            ++pos; // skip {

            TreeNode node = {-1, 0.0f, 0.0f, -1, -1};
            // Parse node fields
            while (pos < json.size() && json[pos] != '}') {
                skip_ws(json, pos);
                if (json[pos] == ',' || json[pos] == '{') { ++pos; continue; }
                if (json[pos] != '"') break;
                std::string key = read_string(json, pos);
                skip_ws(json, pos);
                if (pos < json.size() && json[pos] == ':') ++pos;
                skip_ws(json, pos);

                double val = read_number(json, pos);
                if (key == "feature_idx") node.feature_idx = (int)val;
                else if (key == "threshold") node.threshold = (float)val;
                else if (key == "value")     node.value     = (float)val;
                else if (key == "left")      node.left      = (int)val;
                else if (key == "right")     node.right     = (int)val;

                skip_ws(json, pos);
                if (pos < json.size() && json[pos] == ',') ++pos;
            }
            if (pos < json.size() && json[pos] == '}') ++pos;
            tree.push_back(node);
        }
        if (!tree.empty()) model.trees.push_back(std::move(tree));
    }

    return !model.trees.empty();
}

// ==============================================================================
// Built-in fallback model (used when JSON not available)
// Features: [mean_return, volatility, sharpe, correlation_avg, max_drawdown]
// ==============================================================================
TreeEnsemble build_scoring_model() {
    TreeEnsemble model;
    model.learning_rate = 0.1f;
    model.base_prediction = 0.5f;

    // Tree 1: Split on Sharpe ratio
    std::vector<TreeNode> tree1 = {
        {2, 0.5f, 0.0f, 1, 2},      // root: sharpe <= 0.5?
        {0, 0.08f, 0.0f, 3, 4},     // left: return <= 8%?
        {1, 0.20f, 0.0f, 5, 6},     // right: vol <= 20%?
        {-1, 0, -0.3f, -1, -1},     // leaf: low return, low sharpe
        {-1, 0, 0.1f, -1, -1},      // leaf: ok return, low sharpe
        {-1, 0, 0.4f, -1, -1},      // leaf: high sharpe, low vol
        {-1, 0, 0.2f, -1, -1},      // leaf: high sharpe, high vol
    };

    // Tree 2: Split on volatility
    std::vector<TreeNode> tree2 = {
        {1, 0.25f, 0.0f, 1, 2},     // root: vol <= 25%?
        {3, 0.3f, 0.0f, 3, 4},      // left: corr_avg <= 0.3?
        {-1, 0, -0.2f, -1, -1},     // right leaf: high vol
        {-1, 0, 0.3f, -1, -1},      // leaf: low vol, low corr (best diversifier)
        {-1, 0, 0.1f, -1, -1},      // leaf: low vol, high corr
    };

    // Tree 3: Split on return
    std::vector<TreeNode> tree3 = {
        {0, 0.12f, 0.0f, 1, 2},     // root: return <= 12%?
        {4, 0.15f, 0.0f, 3, 4},     // left: max_drawdown <= 15%?
        {-1, 0, 0.25f, -1, -1},     // right leaf: high return
        {-1, 0, 0.15f, -1, -1},     // leaf: moderate return, low drawdown
        {-1, 0, -0.1f, -1, -1},     // leaf: moderate return, high drawdown
    };

    model.trees = {tree1, tree2, tree3};
    return model;
}

// ==============================================================================
// Score asset combinations in parallel using OpenMP
// ==============================================================================
struct ScoringResult {
    std::vector<float> scores;
    double elapsed_seconds;
    int n_threads;
    double throughput;  // combinations per second
};

ScoringResult score_combinations_parallel(
    const std::vector<std::vector<float>>& feature_matrix,
    const TreeEnsemble& model)
{
    int n = feature_matrix.size();
    std::vector<float> scores(n);

    auto start = std::chrono::high_resolution_clock::now();

    int n_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        n_threads = omp_get_num_threads();
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        scores[i] = model.predict(feature_matrix[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    return {scores, elapsed, n_threads, n / elapsed};
}

// ==============================================================================
// Load model: try JSON file first, fall back to built-in demo.
// Searches common relative paths for the JSON file.
// ==============================================================================
TreeEnsemble load_or_build_model(const std::string& json_path = "") {
    TreeEnsemble model;

    // Try explicit path first
    if (!json_path.empty() && load_model_from_json(json_path, model)) {
        printf("[SCORER] Loaded trained model from %s (%zu trees)\n",
               json_path.c_str(), model.trees.size());
        return model;
    }

    // Try common relative paths
    const char* paths[] = {
        "../data/scorer_model.json",
        "data/scorer_model.json",
        "../portfolio-optimizer/data/scorer_model.json",
    };
    for (auto p : paths) {
        if (load_model_from_json(p, model)) {
            printf("[SCORER] Loaded trained model from %s (%zu trees)\n",
                   p, model.trees.size());
            return model;
        }
    }

    printf("[SCORER] No trained model JSON found — using built-in fallback\n");
    return build_scoring_model();
}

// ==============================================================================
// Generate feature vectors for asset combinations
// n_features controls whether to emit 5 features (built-in model)
// or 7 features (trained model: adds return_spread, vol_ratio).
// ==============================================================================
std::vector<std::vector<float>> generate_combination_features(
    const std::vector<float>& mean_returns,
    const std::vector<float>& volatilities,
    const std::vector<std::vector<float>>& correlation_matrix,
    int n_assets,
    bool extended_features = false)
{
    // Generate all pairs
    std::vector<std::vector<float>> features;
    features.reserve(n_assets * (n_assets - 1) / 2);

    for (int i = 0; i < n_assets; i++) {
        for (int j = i + 1; j < n_assets; j++) {
            float pair_return = 0.5f * (mean_returns[i] + mean_returns[j]);
            float pair_vol = std::sqrt(
                0.25f * volatilities[i] * volatilities[i] +
                0.25f * volatilities[j] * volatilities[j] +
                0.5f * volatilities[i] * volatilities[j] * correlation_matrix[i][j]
            );
            float sharpe = (pair_return - 0.04f) / std::max(pair_vol, 0.001f);

            // Average correlation with other assets
            float avg_corr = 0;
            for (int k = 0; k < n_assets; k++) {
                if (k != i && k != j) {
                    avg_corr += std::abs(correlation_matrix[i][k] + correlation_matrix[j][k]) / 2.0f;
                }
            }
            avg_corr /= std::max(1, n_assets - 2);

            // Simulated max drawdown
            float max_dd = pair_vol * 2.0f;

            if (extended_features) {
                // 7-feature format matching train_scorer.py:
                // [mean_return, volatility, sharpe, correlation, max_drawdown, return_spread, vol_ratio]
                float ret_spread = std::abs(mean_returns[i] - mean_returns[j]);
                float vi = volatilities[i], vj = volatilities[j];
                float vol_ratio = std::min(vi, vj) / std::max(vi, vj + 0.001f);
                features.push_back({pair_return, pair_vol, sharpe, correlation_matrix[i][j],
                                    max_dd, ret_spread, vol_ratio});
            } else {
                // 5-feature format for built-in fallback model
                features.push_back({pair_return, pair_vol, sharpe, avg_corr, max_dd});
            }
        }
    }

    return features;
}

// ==============================================================================
// Main (standalone test)
// ==============================================================================
#ifndef USE_PYBIND11
int main() {
    printf("=== Portfolio Asset Scorer (C++/OpenMP) ===\n\n");

    // Simulate 50 assets
    int n_assets = 50;
    std::vector<float> returns(n_assets), vols(n_assets);
    std::vector<std::vector<float>> corr(n_assets, std::vector<float>(n_assets, 0.0f));

    srand(42);
    for (int i = 0; i < n_assets; i++) {
        returns[i] = 0.05f + (rand() % 100) / 500.0f;  // 5-25% return
        vols[i] = 0.10f + (rand() % 100) / 400.0f;     // 10-35% vol
        corr[i][i] = 1.0f;
        for (int j = i+1; j < n_assets; j++) {
            corr[i][j] = corr[j][i] = 0.1f + (rand() % 60) / 100.0f;
        }
    }

    printf("Generating feature vectors for %d assets...\n", n_assets);
    auto model = load_or_build_model();
    bool extended = (model.trees.size() > 3); // trained model has many trees
    auto features = generate_combination_features(returns, vols, corr, n_assets, extended);
    printf("Generated %zu combinations\n", features.size());
    printf("Model: %zu trees (lr=%.2f)\n\n", model.trees.size(), model.learning_rate);

    auto result = score_combinations_parallel(features, model);

    printf("Results:\n");
    printf("  Combinations scored: %zu\n", features.size());
    printf("  Threads used: %d\n", result.n_threads);
    printf("  Time: %.4f seconds\n", result.elapsed_seconds);
    printf("  Throughput: %.0f combinations/sec\n", result.throughput);

    // Find top scored combination
    int best = std::max_element(result.scores.begin(), result.scores.end()) - result.scores.begin();
    printf("\n  Best score: %.4f (combination #%d)\n", result.scores[best], best);

    return 0;
}
#endif

// ==============================================================================
// Python bindings (pybind11)
// ==============================================================================
#ifdef USE_PYBIND11
PYBIND11_MODULE(scorer, m) {
    m.doc() = "C++/OpenMP Asset Combination Scorer";

    py::class_<ScoringResult>(m, "ScoringResult")
        .def_readonly("scores", &ScoringResult::scores)
        .def_readonly("elapsed_seconds", &ScoringResult::elapsed_seconds)
        .def_readonly("n_threads", &ScoringResult::n_threads)
        .def_readonly("throughput", &ScoringResult::throughput);

    m.def("score_assets", [](
        py::array_t<float> mean_returns_np,
        py::array_t<float> volatilities_np,
        py::array_t<float> corr_matrix_np,
        int n_assets,
        const std::string& model_path
    ) {
        auto mr = mean_returns_np.unchecked<1>();
        auto vol = volatilities_np.unchecked<1>();
        auto corr = corr_matrix_np.unchecked<2>();

        std::vector<float> returns(n_assets), vols(n_assets);
        std::vector<std::vector<float>> correlation(n_assets, std::vector<float>(n_assets));

        for (int i = 0; i < n_assets; i++) {
            returns[i] = mr(i);
            vols[i] = vol(i);
            for (int j = 0; j < n_assets; j++)
                correlation[i][j] = corr(i, j);
        }

        auto model = load_or_build_model(model_path);
        bool extended = (model.trees.size() > 3);
        auto features = generate_combination_features(returns, vols, correlation, n_assets, extended);
        return score_combinations_parallel(features, model);
    }, "Score all asset pair combinations using OpenMP-parallelized tree inference",
    py::arg("mean_returns"), py::arg("volatilities"), py::arg("corr_matrix"),
    py::arg("n_assets"), py::arg("model_path") = "");

    m.def("set_threads", [](int n) { omp_set_num_threads(n); });
    m.def("get_max_threads", []() { return omp_get_max_threads(); });
}
#endif
