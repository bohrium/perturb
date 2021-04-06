//       |       |       |       |       |  
//       |       |       |       |       |  
//       |       |       |       |       |  
//       |       |       |       |       |        
//       |       |       |       |       |        
//     \ | /   \ | /   \ | /   \ | /   \ | /      
//      \|/     \|/     \|/     \|/     \|/       
//       V       V       V       V       V        
// Run:
//      curl -o overfit.c https://gist.githubusercontent.com/anonymous-taylor-series/60ee7ca824e44a9e8f25e69ceb60995e/raw/77fe17c6f2b7b78514439902750a8b5577ee6d77/overfit.c
//      gcc overfit.c -o overfit.o
//      ./overfit.o
//
//       A       A       A       A       A     
//      /|\     /|\     /|\     /|\     /|\   
//     / | \   / | \   / | \   / | \   / | \  
//       |       |       |       |       |           
//       |       |       |       |       |           
//       |       |       |       |       |           
//       |       |       |       |       |           
//       |       |       |       |       |           
//
// The output shows sharp and flat minima both outperforming medium minima: 
//      H=0.01 -> loss=0.17
//      H=0.03 -> loss=0.37
//      H=0.10 -> loss=0.63
//      H=0.30 -> loss=0.48
//      H=1.00 -> loss=0.16

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void main() {
    //------ measurement parameters ------------------------------------------- 
    srand(time(0)); int trials_per_setting=100;

    //------ landscape parameters (l(theta) = H*theta^2/2) -------------------- 
    float noise_scale=1.0;  float Hs[] = {0.01, 0.03, 0.1, 0.3, 1.0, -1};

    //------ optimization parameters ------------------------------------------ 
    int T=1000; float eta=0.01;

    for (float* Hp = Hs; *Hp!=-1; ++Hp) {
        float sum_loss = 0.0; 
        for (int i=0; i!=trials_per_setting; ++i) {
            //------ sample one training point (l_x(theta)=l(theta)+b*theta)---
            float b = noise_scale * ((2.0*rand())/RAND_MAX-1);

            //------ optimization loop, initialized at true minimum -----------
            float theta=0.0;
            for (int t=0; t!=T; ++t) { theta -= eta * ((*Hp)*theta + b); }
            
            //------ compute final testing loss ------------------------------- 
            sum_loss += *Hp * theta*theta/2.0;
        }
        printf("H=%.2f -> loss=%4.2f\n", *Hp, sum_loss/trials_per_setting);
    }
}

