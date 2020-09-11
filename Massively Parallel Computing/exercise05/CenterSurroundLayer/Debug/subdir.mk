################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../center_surround_convolution.cu 

OBJS += \
./center_surround_convolution.o 

CU_DEPS += \
./center_surround_convolution.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.1/cuda/bin/nvcc -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include/TH -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include/THC -I/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.1/cuda/include -I/usr/include/python3.6m -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.1/cuda/bin/nvcc -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include/TH -I/local/var/tmp/env/lib/python3.6/site-packages/torch/include/THC -I/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.1/cuda/include -I/usr/include/python3.6m -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


