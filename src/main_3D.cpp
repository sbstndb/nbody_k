#include <iostream>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include <fstream>
#include <string>
#include <iomanip> // Pour formater la sortie numérique

#include <random>

#include <Kokkos_Core.hpp>


  using integer = int ;
  using real = float ;



// to share with devices!!
template <typename F, typename I>
struct Setting {
	F G = 1.0 ; 
	F dt = 0.01;
	I N = 1024 ; // number of bodies
	I nrepeat = 100 ; // number of repeats of the test
	I frames = 100;  // number of saved frames. Each frame execute nrepeat iterations !

};

template <typename F>
struct Particle {
	F x, y, z;
	F vx, vy, vz;
	F fx, fy, fz;
	F mass ; 
};

// why can't i use template for double ????????? 
//void save_particles_to_file(const Kokkos::View<Particle<double>*>::HostMirror& particles, int n, int  frame_number) {
void save_particles_to_file(const auto& particles, int n, int  frame_number) {

    std::string filename = "frames/frame_" + std::to_string(frame_number) + ".txt";

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Erreur lors de l'ouverture du fichier : " << filename << std::endl;
        return;
    }

    // Enregistrer les particules dans le fichier
    for (int i = 0 ; i < n ; i++) {
        file << particles[i].x << " " << particles[i].y << " " << particles[i].z << " "
             << particles[i].vx << " " << particles[i].vy << " " << particles[i].vz << " "
             << particles[i].fx << " " << particles[i].fy << " " << particles[i].fz << " "
             << particles[i].mass << "\n";
    }
    file.close();
}


int main( int argc, char* argv[] )
{

  Setting<real, integer> setting;

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-n" ) == 0 ) || ( strcmp( argv[ i ], "-N" ) == 0 ) ) {
      setting.N = static_cast<integer>(atoi( argv[++i]));
      printf( "  User N is %d\n", setting.N );
    }
    else if ( ( strcmp( argv[ i ], "-f" ) == 0 ) || ( strcmp( argv[ i ], "-F" ) == 0 ) ) {
      setting.frames = static_cast<integer>(atoi( argv[++i]));
      printf( "  User frames is %d\n", setting.frames );
    }
    else if ( ( strcmp( argv[ i ], "-d" ) == 0 ) || ( strcmp( argv[ i ], "-D" ) == 0 ) ) {
      setting.dt = static_cast<real>(atof( argv[++i]));
      printf( "  User dt is %d\n", setting.dt );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      setting.nrepeat = static_cast<integer>(atoi( argv[ ++i ] ));
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  NBody Options:\n" );
      printf( "  -n (-N) <int>:         number of particles\n" );
      printf( "  -f (-F) <int>:         number of saved frames\n" );      
      printf( "  -d (-D) <int>:         time step value\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }



  Kokkos::initialize( argc, argv );
  {

  // Allocate y, x vectors and Matrix A on device.
  typedef Kokkos::View<Setting<real, integer>*> ViewSetting ; 
  typedef Kokkos::View<Particle<real>*>  ViewVectorType;
  ViewSetting sv("setting", 1);
  ViewVectorType p("p", setting.N);

  // Create host mirrors of device views.
    ViewVectorType::HostMirror h_p = Kokkos::create_mirror_view( p ) ;
    ViewSetting::HostMirror h_sv = Kokkos::create_mirror_view( sv) ;
  h_sv[0] = setting ;     

   std::random_device rd;  // Pour obtenir une graine unique
   std::mt19937 gen(rd()); // Mersenne Twister engine seeded with rd()
   std::uniform_real_distribution<real> dis(0.0, 1.0);

       std::cout << "value : " << dis(gen) << std::endl ;

       h_p(0).x = dis(gen) ; 

  Kokkos::parallel_for("init", Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, setting.N), KOKKOS_LAMBDA (int i){
	h_p(i).x = 0.0;
	h_p(i).y = 0.0;
	h_p(i).z = 0.0;
	h_p(i).vx = 0.0 ;
	h_p(i).vy = 0.0 ; 
	h_p(i).vz = 0.0 ;
	h_p(i).fx = 0.0 ; 
        h_p(i).fy = 0.0 ;
        h_p(i).fz = 0.0 ;
	h_p(i).mass = 1.0;
  });
// i dont know why but i cant atm use dis(gen) in the kokkos kernel in host space 
  for (int i = 0 ; i < setting.N ; i++){
	  h_p(i).x = dis(gen);
	  h_p(i).y = dis(gen);
	  h_p(i).z = dis(gen);
  }

  // Deep copy host views to device views.
    Kokkos::deep_copy(p, h_p);
    Kokkos::deep_copy(sv, h_sv);


  // Timer products.
  Kokkos::Timer timer;
for ( int frame = 0 ; frame < h_sv(0).frames; frame++){
  for ( int repeat = 0; repeat < h_sv(0).nrepeat; repeat++ ) {

      Kokkos::parallel_for("kernel", h_sv(0).N, KOKKOS_LAMBDA (int i){
      real dx , dy , dz;
      real dist, dist3, force;
      p(i).fx = 0.0 ; 
      p(i).fy = 0.0 ; 
      p(i).fz = 0.0 ; 
   
      real eps = 1e-10;
      for (int j = 0 ; j < sv(0).N ; j++){
      	if (i==j)
		continue;
	dx = p(j).x - p(i).x ;

	dy = p(j).y - p(i).y ; 
        dz = p(j).z - p(i).z ;
	dist = sqrt(dx*dx + dy*dy + dz*dz);
	dist3 = dist * dist * dist + eps;
	force = sv(0).G * p(j).mass / dist3 ; 
	p(i).fx += force * dx ; 
	p(i).fy += force * dy ;
	p(i).fz += force * dz ;

      }
      });
      Kokkos::parallel_for("kernel2", h_sv(0).N, KOKKOS_LAMBDA (int i){
        // integration 
	real mass = p(i).mass;
        p(i).vx += sv(0).dt * p(i).fx / mass ;
        p(i).vy += sv(0).dt * p(i).fy / mass ;
        p(i).vz += sv(0).dt * p(i).fz / mass ;
	// integration 2
        p(i).x += p(i).vx * sv(0).dt ;
        p(i).y += p(i).vy * sv(0).dt ;
        p(i).z += p(i).vz * sv(0).dt;
        });

  // here we want to save values in file .txt
  // --> deepcopy to host
  // --> and save in file in folder "result" 
  }
  // save the frame
  Kokkos:deep_copy(h_p, p) ; 
  save_particles_to_file(h_p, h_sv(0).N, frame) ;
  }
// now i purpose a hierarchical parallelism point of view
// we have nested loops (for i for j) but the integration is only for i 
// first method : we use parallel_reduce for them
// second method : we use a second array named fx and fy to store the intermediaire data and then launch a classic 1d parallel loop
  // Calculate time
  Kokkos::fence();
  real time = timer.seconds();
  std::cout << "Elapsed time : " << time << std::endl ; 
  }
  Kokkos::finalize();

  return 0;
}
