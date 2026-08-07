#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace chess_engine_4::inference {

struct InputPlane {
    std::uint64_t mask;
    float value;
};

struct ModelInfo {
    int d_model;
    int depth;
    int history_length;
    int policy_size;
    int maximum_batch_size;
};

struct Outputs {
    std::span<float> policy;
    std::span<float> wdl;
    std::span<float> moves_left;
};

class DenseModel {
public:
    static std::unique_ptr<DenseModel> Load(
        const std::string &path,
        int gpu,
        int maximum_batch_size
    );

    ~DenseModel();
    DenseModel(DenseModel &&) noexcept;
    DenseModel &operator=(DenseModel &&) noexcept;

    DenseModel(const DenseModel &) = delete;
    DenseModel &operator=(const DenseModel &) = delete;

    const ModelInfo &info() const;
    void Evaluate(std::span<const InputPlane> planes, int batch_size, Outputs outputs);

private:
    class Impl;
    explicit DenseModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace chess_engine_4::inference
