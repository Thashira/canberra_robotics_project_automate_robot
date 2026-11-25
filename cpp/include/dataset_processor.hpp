#pragma once

#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace robotics{
    bool validate_steps(
        const pybind11::array_t<int32_t>& steps
    );

    bool validate_alignment(
        const pybind11::array& states,
        const pybind11::array& action,
        const pybind11::array& images
    );

    pybind11::array_t<double> aggregate_states(
        const pybind11::list& chunks
    );

    pybind11::array_t<std::uint8_t> aggregate_images(
        const pybind11::list& chunks
    );
    pybind11::array_t<double> extract_actions(
    const pybind11::list& chunks
);
}