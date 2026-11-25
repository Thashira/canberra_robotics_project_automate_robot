#include <pybind11/pybind11.h>
#include "../include/dataset_processor.hpp"


namespace py = pybind11;

PYBIND11_MODULE(robotics_cpp,m){
    m.doc()="robotics c++ processing backend";
    m.def(
        "validate_steps",
        &robotics::validate_steps,
        "validate sequenstial simulation timestep"
    );
    m.def(
        "validate_alignment",
        &robotics::validate_alignment,
        "Validate dataset array alignment"
    );
    m.def(
        "aggregate_states",
        &robotics::aggregate_states,
        "Aggregate state chunks into a single array"
    );
    m.def(
    "aggregate_images",
    &robotics::aggregate_images,
    "Aggregate RGB image chunks"
    );
    m.def(
    "extract_actions",
    &robotics::extract_actions,
    "Extract action vectors from recorded chunks"
     );
}
