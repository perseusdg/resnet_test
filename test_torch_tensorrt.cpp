#include <torch/script.h>
#include <iostream>
#include <memory>
#include <torch_tensorrt/torch_tensorrt.h>
#include <torch/torch.h>


#define TKDNN_TSTART timespec start, end;                               \
                    clock_gettime(CLOCK_MONOTONIC, &start);            

#define TKDNN_TSTOP_C(col, show)  clock_gettime(CLOCK_MONOTONIC, &end);       \
    double t_ns = ((double)(end.tv_sec - start.tv_sec) * 1.0e9 +       \
                  (double)(end.tv_nsec - start.tv_nsec))/1.0e6;        \
    if(show) std::cout<<col<<"Time:"<<std::setw(16)<<t_ns<<" ms\n"<<COL_END; 

#define TKDNN_TSTOP TKDNN_TSTOP_C(COL_CYANB, TKDNN_VERBOSE)

// Colored output
#define COL_END "\033[0m"

#define COL_RED "\033[31m"
#define COL_GREEN "\033[32m"
#define COL_ORANGE "\033[33m"
#define COL_BLUE "\033[34m"
#define COL_PURPLE "\033[35m"
#define COL_CYAN "\033[36m"

#define COL_REDB "\033[1;31m"
#define COL_GREENB "\033[1;32m"
#define COL_ORANGEB "\033[1;33m"
#define COL_BLUEB "\033[1;34m"
#define COL_PURPLEB "\033[1;35m"
#define COL_CYANB "\033[1;36m"

#define TKDNN_VERBOSE 0



int main(int argc,const char* argv[]){
	torch::jit::script::Module mod;
	try{
		mod = torch::jit::load("resnet101_fp16.ts");
	}catch(const c10::Error& e){
		std::cerr<<"error loading the model \n";
		return -1;
	}

	std::cout<<"Loaded resnet101_fp32"<<std::endl;

	mod.eval();
	mod.to(torch::kCUDA);
	std::vector<double> stats;
	for(int i=0;i<65;i++){
		torch::Tensor in = torch::randn({1,3,224,224},torch::kCUDA).to(torch::kFloat16);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(in);
		torch::cuda::synchronize(-1);
		TKDNN_TSTART
		auto out = mod.forward(inputs);
		TKDNN_TSTOP
		if(i>1){
			stats.push_back(t_ns);
		}
	}
	
  	double min = *std::min_element(stats.begin(), stats.end()); ///BATCH_SIZE;
	double max = *std::max_element(stats.begin(), stats.end()); ///BATCH_SIZE;
    	double mean =0;
    	for(int i=0; i<stats.size(); i++) mean += stats[i]; mean /= stats.size();
    //mean /=BATCH_SIZE;

    	std::cout<<"Min: "<<min<<" ms\n";    
	std::cout<<"Max: "<<max<<" ms\n";    
    	std::cout<<"Avg: "<<mean<<" ms\t"<<1000/(mean)<<" FPS\n"<<COL_END;   

	

}


