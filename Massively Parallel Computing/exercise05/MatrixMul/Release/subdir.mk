################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

OBJS += \
./main.o 

CU_DEPS += \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.1/cuda/bin/nvcc -O3 -ccbin g++-6 -gencode arch=compute_75,code=sm_75  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/graphics/opt/opt_Ubuntu18.04/cuda/toolkit_10.1/cuda/bin/nvcc -O3 -ccbin g++-6 --compile --relocatable-device-code=false -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


