#include "../include/dataset_processor.hpp"
#include <cstring>
#include <stdexcept>

namespace robotics{
//newfunction defined
         bool validate_steps(const pybind11::array_t<int32_t>& steps)
              {
                if (steps.ndim() !=1){
                    return false;
                }

                auto data = steps.unchecked<1>();

                for (pybind11::ssize_t i = 0;i < data.shape(0) ; ++i){
                    if (data(i) != i){
                        return false;
                    }
                }

                return true;
              };
//newfunction defined     
         bool validate_alignment(const pybind11::array& states,const pybind11::array& actions,const pybind11::array& images){
            if (states.ndim()!=2 or actions.ndim()!=2 or images.ndim()!=4){
                return false;
            }

            const auto n_states = states.shape(0);
            const auto n_actions =actions.shape(0);
            const auto n_images = images.shape(0);

            return (n_states==n_images and n_states==n_actions+1);
         };   
//newfunction defined
         pybind11::array_t<double> aggregate_states(const pybind11::list& chunks){
            namespace py = pybind11;
              
            py::ssize_t total_row=0;
            py::ssize_t columns=71;
            
            for (auto item : chunks){
                auto chunk= py::cast<py::array>(item);

            if (chunk.ndim()!=2 or chunk.shape(1)!= columns or !chunk.dtype().is(py::dtype::of<double>()) or !(chunk.flags() and py::array::c_style)){
                throw std::runtime_error("Invalid state chunk dumber");
            }
            total_row+=chunk.shape(0);
            }

            py::array_t<double> results({total_row,columns});
            double* destination=results.mutable_data();
            py::ssize_t offset=0;

            for (auto item:chunks){
                auto chunk =py::cast<py::array>(item);
                std::memcpy(destination+offset,chunk.data(),chunk.nbytes());
                offset+=chunk.size();
            }
            return results;
         };
 //newfunction defined
         pybind11::array_t<std::uint8_t> aggregate_images(const pybind11::list& chunks){
            namespace py = pybind11;

            if (chunks.empty()){
                throw std::runtime_error("Empty image chunk");
            }
            auto first_chunk=py::cast<py::array>(chunks[0]);

            if (first_chunk.ndim()!=4){
                throw std::runtime_error("Expected 4D image chunk");
            }

            const auto height= first_chunk.shape(1);
            const auto width = first_chunk.shape(2);
            const auto channels = first_chunk.shape(3);

            py::ssize_t total_frames=0;

            for( auto item:chunks){
                auto chunk=py::cast<py::array>(item);
                if(chunk.ndim()!=4 or chunk.shape(1)!=height or chunk.shape(2)!=width or chunk.shape(3)!=channels or !chunk.dtype().is(py::dtype::of<std::uint8_t>()) or !(chunk.flags() and py::array::c_style)){
                    throw std::runtime_error("Invalid image chunk");
                }
                total_frames+=chunk.shape(0);
            }
            
            py::array_t<std::uint8_t> results({total_frames,height,width,channels});

            auto* destination = results.mutable_data();

            py::ssize_t offset=0;

            for(auto item:chunks){
                auto chunk=py::cast<py::array>(item);
                if (chunk.size() == 0)
                    {
                        continue;
                    }
                std::memcpy(destination+offset,chunk.data(),chunk.nbytes());
                offset+=chunk.size();
            }
             
            return results;


         }
//newfunction defined
         pybind11::array_t<double> extract_actions(const pybind11::list& chunks)
        {
            namespace py = pybind11;

            py::ssize_t total_actions = 0;

            const py::ssize_t action_dim = 7;

            // Calculate total number of actions
            for (auto item : chunks)
            {
                total_actions += py::len(item);
            }

            // Allocate output array
            py::array_t<double> result({
                total_actions,
                action_dim
            });

            double* destination = result.mutable_data();

            py::ssize_t offset = 0;

            // Extract action vectors
            for (auto item : chunks)
            {
                auto chunk = py::reinterpret_borrow<py::iterable>(
                    item
                );

                for (auto entry : chunk)
                {
                    auto info = py::cast<py::dict>(entry);

                    auto action = py::array_t<
                        double,
                        py::array::c_style |
                        py::array::forcecast
                    >::ensure(info["actions"]);

                    if (!action ||
                        action.ndim() != 1 ||
                        action.shape(0) != action_dim)
                    {
                        throw std::runtime_error(
                            "Invalid action vector"
                        );
                    }

                    std::memcpy(
                        destination + offset,
                        action.data(),
                        action.nbytes()
                    );

                    offset += action.size();
                }
            }

            return result;
        }
}




