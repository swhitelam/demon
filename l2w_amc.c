
//flag: objective altered

#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>

using namespace std;

char st[256];
long seed;

//mmm, pi
const double pi=4.0*atan(1.0);

//production
const int q_production_run=1;

//size- and time parameters
const int L_x=32;
const int L_y=32;
const int mc_sweeps=1000;

//training
const int epoch_reset=2000;
const int epoch_report=2000;
const int swarm_epoch_report=20000;

int number_of_trajectories=1000;
int number_of_trajectories_swarm=100; //for visualization

//size- and time parameters (fixed)
const int number_of_sites=L_x*L_y;
const int trajectory_length=mc_sweeps*number_of_sites;
const int net_actions=1000;
const int net_step=trajectory_length/net_actions;
const int report_step=trajectory_length/net_actions;
const int picture_step=trajectory_length/10;

//net parameters
const int depth=4; //stem of mushroom net, >=1 (so two-later net at least)
const int width=4;
const int width_final=10;

const int number_of_inputs=2;
const int number_of_outputs=2;
const int number_of_net_parameters=number_of_inputs*width+width*width*(depth-1)+width*width_final+width_final*number_of_outputs+depth*width+width_final+number_of_outputs;

//two single-layer nets
//const int number_of_net_parameters=2*(number_of_inputs*width+2*width+1);

//aMC
int n_scale=50;
int q_layer_norm=1;
double epsilon=0.01;
double sigma_mutate_zero=0.1;
double sigma_mutate=sigma_mutate_zero;
double sigma_mutate_initial=0.2;

int q_ok=0;
int consec_rejections;

long n_reset=0;

double np;
double phi;
double current_mean_mag;

//endpoint transformation
int q_shear;
double shear[2][2];

//registers
double mutation[number_of_net_parameters];
double mean_mutation[number_of_net_parameters];
double net_parameters[number_of_net_parameters];
double net_parameters_holder[number_of_net_parameters];

//vars int
int tau_int=0;
int generation=0;
int show_picture=1;
int record_trajectory=1;
int spin[number_of_sites];
int neighbors[number_of_sites][4];

//vars double
double tee;
double ising_field;

double tee_final=0.65;
double tee_initial=0.65;
double ising_field_final=1.0;
double ising_field_initial=-1.0;

double tee_critical=1.0/(0.5*log(1.0+sqrt(2.0)));

//vars double (fixed)
double max_ep;
double min_ep;
double mean_ep;
double entprod;
double activity;
double mean_mag;
double tau=0.0;
double magnetization;
double ising_jay=1.0; //fixed

//functions void
void ga(void);
void learn(void);
void mc_step(void);
void read_net(void);
void store_net(void);
void initialize(void);
void output_net(void);
void mutate_net(void);
void restore_net(void);
void jobcomplete(int i);
void start_picture(void);
void initialize_net(void);
void run_trajectory(void);
void scale_mutations(void);
void calculate_shear(void);
void reset_module(int gen1);
void run_net(int step_number);
void initialize_lattice(void);
void make_picture_render(void);
void make_picture_showpage(void);
void run_trajectory_average(void);
void output_generational_data1(int gen1);
void output_generational_data2(int gen1);
void output_generational_data3(int gen1);
void add_picture_lattice(int step_number);
void output_trajectory_data(int step_number);
void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max);
void plot_function_swarm(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max);


//functions int
int node_x_coord(int i);
int node_y_coord(int i);

//functions double
double test_phi(void);
double lattice_nrg(void);
double gauss_rv(double sigma);
double delta_nrg(int chosen_site);

int main(void){
  
//RN generator, lattice, net
initialize();

//aMC
//learn();

//GA
ga();

}

void initialize(void){

//clean up
sprintf(st,"rm report_*");
cout << st << endl;
cout << endl;
system(st);

sprintf(st,"rm net_out_gen_%d.dat",generation-2);
cout << st << endl;
cout << endl;
system(st);

ifstream infile0("input_parameters.dat", ios::in);
while (!infile0.eof ()){infile0 >> seed >> generation;}

//seed RN generator
srand48(seed);

//initialize net
if(generation==0){initialize_net();}
else{read_net();}

//set up lattice
initialize_lattice();

}


void initialize_lattice(void){

int i;
int x1,y1;

for(i=0;i<number_of_sites;i++){

x1=node_x_coord(i);
y1=node_y_coord(i);

//left
//if(x1==0){neighbors[i][0]=-1;}
if(x1==0){neighbors[i][0]=i+(L_x-1);}
else{neighbors[i][0]=i-1;}

//right
//if(x1==L_x-1){neighbors[i][1]=-1;}
if(x1==L_x-1){neighbors[i][1]=i-(L_x-1);}
else{neighbors[i][1]=i+1;}

//down
//if(y1==0){neighbors[i][2]=-1;}
if(y1==0){neighbors[i][2]=i+L_x*(L_x-1);}
else{neighbors[i][2]=i-L_x;}

//up
//if(y1==L_y-1){neighbors[i][3]=-1;}
if(y1==L_y-1){neighbors[i][3]=i-L_x*(L_x-1);}
else{neighbors[i][3]=i+L_x;}

}

/*
for(i=0;i<number_of_sites;i++){
cout << "node " << i << " (x,y) " << node_x_coord(i) << " " << node_y_coord(i) << endl;
for(int j=0;j<4;j++){
cout << " neighbor " << neighbors[i][j] << " (x,y) " << node_x_coord(neighbors[i][j]) << " " << node_y_coord(neighbors[i][j]) << endl;
}}
exit(2);
*/

}

int node_x_coord(int i){
 
 int q= (i % L_x);
 
 return (q);
 
}

int node_y_coord(int i){
 
 int q= (i / L_x);
 
 return (q);
 
}


double gauss_rv(double sigma){

double r1,r2;
double g1;
double two_pi = 2.0*pi;

r1=drand48();
r2=drand48();

g1=sqrt(-2.0*log(r1))*sigma*cos(two_pi*r2);

return (g1);

}


double delta_nrg(int chosen_site){

int i;

double s2;
double de=0.0;
double s1=1.0*spin[chosen_site];

//coupling
for(i=0;i<4;i++){

s2=1.0*spin[neighbors[chosen_site][i]];
de+=2.0*ising_jay*s1*s2;

}

//field
de+=2.0*ising_field*s1;

return (de);

}

void run_trajectory(void){

int i;
double e1,e2;

//start picture
start_picture();

//zero counters
tau=0.0;
tau_int=0;
entprod=0.0;
activity=0.0;
magnetization=0.0;

//endpoints
calculate_shear();

//lattice
magnetization=0.0;
for(i=0;i<number_of_sites;i++){spin[i]=-1;magnetization-=1.0;}
magnetization*=1.0/(1.0*number_of_sites);

//set initial conditions
run_net(0);
e1=lattice_nrg();

//run traj
for(i=0;i<trajectory_length;i++){

run_net(i);
output_trajectory_data(i);
mc_step();
add_picture_lattice(i);

tau_int++;
tau+=1.0/(1.0*trajectory_length);

}

//endpoint
tee=tee_final;
ising_field=ising_field_final;
e2=lattice_nrg();
entprod+=(e2-e1)/(tee_final);
output_trajectory_data(trajectory_length);


}

void mc_step(void){

int chosen_site = (int) (drand48()*number_of_sites);

double de=delta_nrg(chosen_site);

double prob1=drand48();

double glauber=exp(-de/tee);
glauber=glauber/(glauber+1.0);

if(prob1<glauber){

activity+=1.0/(1.0*trajectory_length);
entprod-=de/tee;
magnetization-=2.0*spin[chosen_site]/(1.0*number_of_sites);
spin[chosen_site]=-spin[chosen_site];

}

}


void output_trajectory_data(int step_number){

if(record_trajectory==1){
if((step_number % report_step==0) || (step_number==trajectory_length)){

sprintf(st,"report_mag_gen_%d.dat",generation);
ofstream out1(st,ios::app);
sprintf(st,"report_entprod_gen_%d.dat",generation);
ofstream out2(st,ios::app);
sprintf(st,"report_field_gen_%d.dat",generation);
ofstream out3(st,ios::app);
sprintf(st,"report_tee_gen_%d.dat",generation);
ofstream out4(st,ios::app);
sprintf(st,"report_tee_field_gen_%d.dat",generation);
ofstream out5(st,ios::app);
sprintf(st,"report_mag_field_gen_%d.dat",generation);
ofstream out6(st,ios::app);
sprintf(st,"report_tee_mag_gen_%d.dat",generation);
ofstream out7(st,ios::app);

out1 << tau << " " << magnetization << endl;
if(tau>0){out2 << tau << " " << entprod << endl;}
out3 << tau << " " << ising_field << endl;
out4 << tau << " " << tee << endl;
out5 << tee << " " << ising_field << endl;
out6 << magnetization << " " << ising_field << endl;
out7 << tee << " " << magnetization << endl;

}}

}


void plot_function(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max){

    
//sprintf(st,"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/config.asy ."); system(st);
//sprintf(st,"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/graph_routines.asy ."); system(st);
const char *varname2="tee_field";

 //output file
 sprintf(st,"report_%s.asy",varname);
 ofstream output_interface_asy(st,ios::out);
 
 //write output file
 output_interface_asy << "import graph;" << endl;
 output_interface_asy << "import stats;"<< endl;
 
 output_interface_asy << "from \"graph_routines.asy\" access *;"<< endl;
 
 output_interface_asy << "picture p2;"<< endl;
 output_interface_asy << "defaultpen(1.5);"<< endl;
 
 output_interface_asy << "real ymin=" << y_min << ";"<< endl;
 output_interface_asy << "real ymax=" << y_max << ";"<< endl;
 
 output_interface_asy << "real xmin=" << x_min << ";"<< endl;
 output_interface_asy << "real xmax=" << x_max << ";"<< endl;
 
 output_interface_asy << "size(p2,400,400,IgnoreAspect);"<< endl;
 output_interface_asy << "scale(p2,Linear(true),Linear(true));"<< endl;
 
 
 //void simplot_symbol(picture p, string filename,string name,pen pn,int poly,real a1,real a2,real a3,real s1)

sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,generation,0.0,0.0,1.0,1.7);
output_interface_asy << st << endl;

if(varname==varname2){
sprintf(st,"simplot_simple(p2,\"data/line.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.0,0.0,1.0,1.7);
output_interface_asy << st << endl;

}


 output_interface_asy << "xlimits(p2, xmin, xmax, true);"<< endl;
 output_interface_asy << "ylimits(p2, ymin, ymax, true);"<< endl;
 
 sprintf(st,"%s",xlabel);
 output_interface_asy << "xaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),BottomTop,LeftTicks(Label(fontsize(30)),new real[]{" << x_min <<"," << x_max <<"})); "<< endl;
 sprintf(st,"%s",ylabel);
 output_interface_asy << "yaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),LeftRight,RightTicks(Label(fontsize(30)),new real[]{"<< y_min << "," << y_max <<"})); "<< endl;
 
 
 output_interface_asy << "scale(Linear(true),Linear(true)); "<< endl;
 output_interface_asy << "add(p2.fit(250,250),(0,0),S);"<< endl;

if(q_production_run==0){
sprintf(st,"/usr/local/texlive/2017/bin/x86_64-darwin/asy report_%s.asy",varname);
system(st);

sprintf(st,"open report_%s.eps",varname);
system(st);
}
 
 
}


void plot_function_swarm(const char *varname, const char *xlabel, double x_min, double x_max, const char *ylabel, double y_min, double y_max){

int i;

//sprintf(st,"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/config.asy ."); system(st);
//sprintf(st,"cp /Users/swhitelam/asymptote/bin/useful_files/crystal_growth_poisoning_simple/graph_routines.asy ."); system(st);
const char *varname2="tee_field";

 //output file
 sprintf(st,"report_%s_swarm.asy",varname);
 ofstream output_interface_asy(st,ios::out);
 
 //write output file
 output_interface_asy << "import graph;" << endl;
 output_interface_asy << "import stats;"<< endl;
 
 output_interface_asy << "from \"graph_routines.asy\" access *;"<< endl;
 
 output_interface_asy << "picture p2;"<< endl;
 output_interface_asy << "defaultpen(1.5);"<< endl;
 
 output_interface_asy << "real ymin=" << y_min << ";"<< endl;
 output_interface_asy << "real ymax=" << y_max << ";"<< endl;
 
 output_interface_asy << "real xmin=" << x_min << ";"<< endl;
 output_interface_asy << "real xmax=" << x_max << ";"<< endl;
 
 output_interface_asy << "size(p2,400,400,IgnoreAspect);"<< endl;
 output_interface_asy << "scale(p2,Linear(true),Linear(true));"<< endl;
 
 
 //void simplot_symbol(picture p, string filename,string name,pen pn,int poly,real a1,real a2,real a3,real s1)

for(i=0;i<number_of_trajectories_swarm;i++){
sprintf(st,"simplot_simple(p2,\"report_%s_gen_%d_traj_%d.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",varname,generation,i,0.0,0.0,1.0,0.25);
output_interface_asy << st << endl;
}

if(varname==varname2){
sprintf(st,"simplot_simple(p2,\"data/line.dat\",\"\\Large $$\",pen1,%g,%g,%g,%g);",0.0,0.0,1.0,1.7);
output_interface_asy << st << endl;

}


 output_interface_asy << "xlimits(p2, xmin, xmax, true);"<< endl;
 output_interface_asy << "ylimits(p2, ymin, ymax, true);"<< endl;
 
 sprintf(st,"%s",xlabel);
 output_interface_asy << "xaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),BottomTop,LeftTicks(Label(fontsize(30)),new real[]{" << x_min <<"," << x_max <<"})); "<< endl;
 sprintf(st,"%s",ylabel);
 output_interface_asy << "yaxis(p2,Label(\"$"<<st<<"$\",fontsize(30pt)),LeftRight,RightTicks(Label(fontsize(30)),new real[]{"<< y_min << "," << y_max <<"})); "<< endl;
 
 
 output_interface_asy << "scale(Linear(true),Linear(true)); "<< endl;
 output_interface_asy << "add(p2.fit(250,250),(0,0),S);"<< endl;

if(q_production_run==0){
sprintf(st,"/usr/local/texlive/2017/bin/x86_64-darwin/asy report_%s_swarm.asy",varname);
system(st);

sprintf(st,"open report_%s_swarm.eps",varname);
system(st);
}
 
 
}

 void add_picture_lattice(int step_number){
 
if(show_picture==1){
if(((step_number % picture_step==0) && (step_number>0)) || (step_number == trajectory_length-1)){

 int i;
 
 double x1=10;
 double y1=10;
  
 ofstream output_config("report_picture.ps",ios::app);
 
 double pic_scale=5;
 output_config << pic_scale << " " <<  pic_scale <<" scale" << endl;
  
  output_config << "  0 0 0 setrgbcolor" << endl;
 output_config << "  0.1 setlinewidth" << endl;
 output_config << "  newpath" << endl;
 output_config << "  " << x1 << " " << y1 << " moveto" << endl;
 output_config << "  " << x1+L_x << " " << y1 << " lineto" << endl;
 output_config << "  " << x1+L_x << " " << y1+L_y << " lineto" << endl;
 output_config << "  " << x1 << " " << y1+L_y << " lineto" << endl;
  output_config << "  " << x1 << " " << y1 << " lineto" << endl;
 output_config << "  closepath stroke" << endl;
 
  output_config << "  " << x1 << " " << y1 << " moveto" << endl;
 

 for(i=0;i<number_of_sites;i++){
   
 if((node_x_coord(i)==0) && (node_y_coord(i)>0)){output_config << -(L_x-1) << " " << 1  << " translate " << endl;}
 else{output_config << 1 << " " << 0  << " translate " << endl;}
   

 if(spin[i]>0){output_config << 0 << " " << 0  << " " << 1  << " csquare" << endl;}

 }
 
 make_picture_showpage();
 
 }}
 
 }
 


 void make_picture_showpage(void){
 
ofstream output_config("report_picture.ps",ios::app);
output_config << " showpage " << endl;

 
 }
 
 void make_picture_render(void){
 
 if(show_picture==1){

 sprintf(st,"ps2pdf14 report_picture.ps");
 system(st);
 
 sprintf(st,"open report_picture.pdf");
 system(st);
 
 }
 }
 
 
void start_picture(void){

if(show_picture==1){

 double x1=10;
 double y1=10;

ofstream output_config("report_picture.ps",ios::app);
 
output_config << " /Times-Roman findfont " << endl;
output_config << " 20 scalefont  " << endl;
output_config << "setfont " << endl;

 //define square
 output_config << "  /csquare {" << endl;
 output_config << "  newpath" << endl;
 output_config << "  " << x1-1 << " " << y1 << " moveto" << endl;
 output_config << "  0 1 rlineto" << endl;
 output_config << "  1 0 rlineto" << endl;
 output_config << "  0 -1 rlineto" << endl;
 output_config << "  closepath" << endl;
 output_config << "  setrgbcolor" << endl;
  output_config << "  fill" << endl;
 output_config << " } def" << endl;
 
}}



void read_net(void){

int i;

sprintf(st,"net_in_gen_%d.dat",generation);
ifstream infile(st, ios::in);

for(i=0;i<number_of_net_parameters;i++){infile >> net_parameters[i];}

}

void output_net(void){

int i;

//parameter file
sprintf(st,"net_out_gen_%d.dat",generation);
ofstream out_net(st,ios::out);

for(i=0;i<number_of_net_parameters;i++){out_net << net_parameters[i] << " ";}

}


void store_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters_holder[i]=net_parameters[i];}

}

void mutate_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){mutation[i]=mean_mutation[i]+gauss_rv(sigma_mutate);}
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]+=mutation[i];}

}


void restore_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=net_parameters_holder[i];}

}

void initialize_net(void){

int i;
for(i=0;i<number_of_net_parameters;i++){net_parameters[i]=gauss_rv(sigma_mutate_initial);}

}


void run_net(int step_number){
if(step_number % net_step==0){

int pid=0;
int i,j,k;

double mu=0.0;
double sigma=0.0;
double delta=1e-4;

double inputs[number_of_inputs];
double hidden_node[width][depth];
double outputs[number_of_outputs];
double hidden_node_final[width_final];

//load inputs
inputs[0]=tau;
inputs[1]=magnetization;

//surface layer
for(i=0;i<width;i++){
hidden_node[i][0]=net_parameters[pid];pid++;
for(j=0;j<number_of_inputs;j++){hidden_node[i][0]+=inputs[j]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
if(q_layer_norm==1){
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][0];sigma+=hidden_node[i][0]*hidden_node[i][0];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][0]=(hidden_node[i][0]-mu)/sigma;}
}

//activation
for(i=0;i<width;i++){hidden_node[i][0]=tanh(hidden_node[i][0]);}


//stem layers
for(j=1;j<depth;j++){
for(i=0;i<width;i++){
hidden_node[i][j]=net_parameters[pid];pid++;
for(k=0;k<width;k++){hidden_node[i][j]+=hidden_node[k][j-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
if(q_layer_norm==1){
mu=0.0;sigma=0.0;
for(i=0;i<width;i++){mu+=hidden_node[i][j];sigma+=hidden_node[i][j]*hidden_node[i][j];}
mu=mu/width;sigma=sigma/width;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width;i++){hidden_node[i][j]=(hidden_node[i][j]-mu)/sigma;}
}

//activation
for(i=0;i<width;i++){hidden_node[i][j]=tanh(hidden_node[i][j]);}

}

//final layer
for(i=0;i<width_final;i++){
hidden_node_final[i]=net_parameters[pid];pid++;
for(j=0;j<width;j++){hidden_node_final[i]+=hidden_node[j][depth-1]*net_parameters[pid];pid++;}
}

//layer norm (pre-activation)
if(q_layer_norm==1){
mu=0.0;sigma=0.0;
for(i=0;i<width_final;i++){mu+=hidden_node_final[i];sigma+=hidden_node_final[i]*hidden_node_final[i];}
mu=mu/width_final;sigma=sigma/width_final;
sigma=sqrt(sigma-mu*mu)+delta;
for(i=0;i<width_final;i++){hidden_node_final[i]=(hidden_node_final[i]-mu)/sigma;}
}

//activation
for(i=0;i<width_final;i++){hidden_node_final[i]=tanh(hidden_node_final[i]);}

//outputs
for(i=0;i<number_of_outputs;i++){
outputs[i]=0*net_parameters[pid];pid++;
for(j=0;j<width_final;j++){outputs[i]+=hidden_node_final[j]*net_parameters[pid];pid++;}
}


//temperature
if(q_shear==0){tee=outputs[0];}
else{tee=outputs[0]+(1.0-tau)*shear[0][0]+tau*shear[0][1];
if(tee<1e-3){tee=1e-3;}}

//magnetic field
if(q_shear==0){ising_field=outputs[1];}
else{ising_field=outputs[1]+(1.0-tau)*shear[1][0]+tau*shear[1][1];}

//+4.0*tau*(1.0-tau)*(10-0.65)+0.65;
//+2.0*tau-1.0;

}}


void learn(void){

//establish initial phi
show_picture=0;
record_trajectory=0;
run_trajectory_average();
phi=np;
current_mean_mag=mean_mag;
q_ok=1;

while(2>1){

reset_module(generation);
output_generational_data1(generation);
output_generational_data2(generation);
output_generational_data3(generation);

store_net();
mutate_net();
run_trajectory_average();

if(np<=phi){q_ok=1;phi=np;current_mean_mag=mean_mag;}
else{q_ok=0;restore_net();}

scale_mutations();
generation++;

}
}


void jobcomplete(int i){

 //sprintf(st,"rm jobcomplete.dat");
 //system(st);
 
 sprintf(st,"jobcomplete.dat");
 ofstream output_job(st,ios::out);
 output_job << i << endl;
 output_job.close();

}


void scale_mutations(void){

int i;

double x1=0.0;
double m1=0.0;

if(q_ok==1){

consec_rejections=0;

//mutations
for(i=0;i<number_of_net_parameters;i++){

x1=mutation[i];
m1=mean_mutation[i];

mean_mutation[i]+=epsilon*(x1-m1);

}}
else{

consec_rejections++;

if(consec_rejections>=n_scale){

n_reset++;
consec_rejections=0;
sigma_mutate*=0.95;

for(i=0;i<number_of_net_parameters;i++){mean_mutation[i]=0;}

}}


}


double lattice_nrg(void){

int i;
int j;
int n1;
double s1,s2;
double nrg1=0.0;

for(i=0;i<number_of_sites;i++){

s1=spin[i];
nrg1-=ising_field*s1;

for(j=0;j<4;j++){

n1=neighbors[i][j];
s2=spin[n1];
nrg1-=0.5*s1*s2;

}
}

return (nrg1);

}


void run_trajectory_average(void){

int i;

mean_ep=0.0;
mean_mag=0.0;

double k_m=1.0;
double k_sigma=1e-4;
double n1=1.0/(1.0*number_of_trajectories);

for(i=0;i<number_of_trajectories;i++){

//run trajectory
run_trajectory();

//averages
mean_ep+=n1*entprod;
mean_mag+=n1*magnetization;

//extrema
if(i==0){max_ep=entprod;min_ep=entprod;}
else{
if(entprod>max_ep){max_ep=entprod;}
if(entprod<min_ep){min_ep=entprod;}
}}

//flag
//construct phi
//np=k_m*fabs(mean_mag-1.0)+k_sigma*fabs(mean_ep);
np=k_m*fabs(mean_mag-1.0)+k_sigma*mean_ep;


}


void output_generational_data1(int gen1){
if((q_ok==1) || (gen1 % epoch_report==0)){

sprintf(st,"report_mean_mag.dat");
ofstream out1(st,ios::app);

sprintf(st,"report_mean_ep.dat");
ofstream out2(st,ios::app);

sprintf(st,"report_max_ep.dat");
ofstream out3(st,ios::app);

sprintf(st,"report_min_ep.dat");
ofstream out4(st,ios::app);

sprintf(st,"report_phi.dat");
ofstream out5(st,ios::app);

out1 << generation << " " << mean_mag << endl;
out2 << generation << " " << mean_ep << endl;
out3 << generation << " " << max_ep << endl;
out4 << generation << " " << min_ep << endl;
out5 << generation << " " << phi << endl;

if((q_production_run==0) && (q_ok==1)){
cout << generation << " phi = " << np << " <m>= " << mean_mag << " <sigma>= " << mean_ep << endl;
cout << generation <<  " sigma_max " << max_ep << " sigma_min " << min_ep << endl;
cout << endl;
}

}}


void output_generational_data2(int gen1){
if((gen1 % epoch_report==0) && (gen1>-1)){

//output net
output_net();

//output picture
record_trajectory=1;
show_picture=1;
run_trajectory();
show_picture=0;
record_trajectory=0;

//plots
//plot_function("mag","t/t_0",0,1,"m",-1.1,1.1);
//plot_function("field","t/t_0",0,1,"h",-2.1,2.1);
//plot_function("tee","t/t_0",0,1,"T",0,10);

plot_function("tee_field","T",0,10,"h",-1.5,1.5);
plot_function("mag_field","m",-1.1,1.1,"h",-1.5,1.5);
plot_function("tee_mag","T",0,10,"m",-1.1,1.1);

}}

void output_generational_data3(int gen1){
if((gen1 % swarm_epoch_report==0) && (gen1>0)){

int i;

//output picture
record_trajectory=1;
show_picture=1;

for(i=0;i<number_of_trajectories_swarm;i++){

run_trajectory();

//sprintf(st,"cp report_mag_gen_%d.dat report_mag_gen_%d_traj_%d.dat",generation,generation,i);system(st);
//sprintf(st,"rm report_mag_gen_%d.dat",generation);system(st);

//sprintf(st,"cp report_entprod_gen_%d.dat report_entprod_gen_%d_traj_%d.dat",generation,generation,i);system(st);
//sprintf(st,"rm report_entprod_gen_%d.dat",generation);system(st);

//sprintf(st,"cp report_field_gen_%d.dat report_field_gen_%d_traj_%d.dat",generation,generation,i);system(st);
//sprintf(st,"rm report_field_gen_%d.dat",generation);system(st);

//sprintf(st,"cp report_tee_gen_%d.dat report_tee_gen_%d_traj_%d.dat",generation,generation,i);system(st);
//sprintf(st,"rm report_tee_gen_%d.dat",generation);system(st);

//sprintf(st,"cp report_tee_field_gen_%d.dat report_tee_field_gen_%d_traj_%d.dat",generation,generation,i);system(st);
//sprintf(st,"rm report_tee_field_gen_%d.dat",generation);system(st);

sprintf(st,"cp report_mag_field_gen_%d.dat report_mag_field_gen_%d_traj_%d.dat",generation,generation,i);system(st);
sprintf(st,"rm report_mag_field_gen_%d.dat",generation);system(st);

sprintf(st,"cp report_tee_mag_gen_%d.dat report_tee_mag_gen_%d_traj_%d.dat",generation,generation,i);system(st);
sprintf(st,"rm report_tee_mag_gen_%d.dat",generation);system(st);

}

show_picture=0;
record_trajectory=0;

//plot_function_swarm("tee_field","T",0,10,"h",-1.5,1.5);
plot_function_swarm("mag_field","m",-1.1,1.1,"h",-1.5,1.5);
plot_function_swarm("tee_mag","T",0,10,"m",-1.1,1.1);

}}


void calculate_shear(void){

q_shear=0;

//initial conditions
tau=0.0;
magnetization=-1.0;
run_net(0);
shear[0][0]=tee_initial-tee;
shear[1][0]=ising_field_initial-ising_field;

//final conditions
tau=1.0;
magnetization=1.0;
run_net(0);
shear[0][1]=tee_final-tee;
shear[1][1]=ising_field_final-ising_field;

//cout << "shear " << shear[0][0] << " " << shear[1][0] << endl;
//cout << " shear " << shear[0][1] << " " << shear[1][1] << endl;


/*
//test
q_shear=1;
tau=0.0;
run_net(0);
cout << tee << " " << ising_field << endl;

tau=1.0;
run_net(0);
cout << tee << " " << ising_field << endl;
exit(2);
*/

//turn on shear
q_shear=1;

//initial conditions
tau=0.0;
magnetization=0.0;

}

void reset_module(int gen1){
if((gen1>epoch_reset) && (current_mean_mag<-0.9)){

int i;

cout << gen1 << " " << current_mean_mag << " resetting " << endl;

//counter
generation=0;

//clean up
sprintf(st,"rm report_*");
cout << st << endl;
cout << endl;
system(st);

sprintf(st,"rm net_out_gen_*");
cout << st << endl;
cout << endl;
system(st);

//reset net
initialize_net();

//reset aMC
for(i=0;i<number_of_net_parameters;i++){mean_mutation[i]=0.0;}

n_reset=0;
consec_rejections=0;
sigma_mutate=sigma_mutate_zero;

//establish initial phi
show_picture=0;
record_trajectory=0;
run_trajectory_average();
phi=np;
current_mean_mag=mean_mag;
q_ok=1;


}}


void ga(void){

record_trajectory=0;
show_picture=0;

//mutate net
if(generation>0){mutate_net();};

//calculate order parameter
run_trajectory_average();

//output evolutionary order parameter
sprintf(st,"report_phi_gen_%d.dat",generation);
ofstream output_phi(st,ios::out);
output_phi << np << endl;

//output other order parameters
sprintf(st,"report_order_parameters_gen_%d.dat",generation);
ofstream output_op(st,ios::out);
output_op << mean_mag << " " << mean_ep << " " << min_ep << " " << max_ep << endl;

//output net
output_net();

//output data
record_trajectory=1;
show_picture=1;
run_trajectory();

//jobcomplete
jobcomplete(1);

if(q_production_run==0){

//plots
plot_function("mag","t/t_0",0,1,"m",-1.1,1.1);
plot_function("entprod","t/t_0",0,1,"\\sigma",-1,1);
plot_function("field","t/t_0",0,1,"h",-2.1,2.1);
plot_function("tee","t/t_0",0,1,"T",0,10);
plot_function("tee_field","T",0,10,"h",-2.1,2.1);

//make_picture
make_picture_render();
}

}
