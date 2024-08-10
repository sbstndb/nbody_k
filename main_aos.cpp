#include <iostream>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>


#include <Kokkos_Core.hpp>


const double dt = 0.01 ; 
const double G = 1.0;


struct Particle {
	double x, y;
	double vx, vy;
	double mass ; 
};


int main( int argc, char* argv[] )
{
  int N = 1024; // number off bodies
  int nrepeat = 100;  // number of repeats of the test


  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-n" ) == 0 ) || ( strcmp( argv[ i ], "-N" ) == 0 ) ) {
      N = atoi( argv[++i]);
      printf( "  User N is %d\n", N );
    }
    else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) {
      nrepeat = atoi( argv[ ++i ] );
    }
    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      printf( "  y^T*A*x Options:\n" );
      printf( "  -n (-N) <int>:      number of particles\n" );
      printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
      printf( "  -help (-h):            print this message\n\n" );
      exit( 1 );
    }
  }


  Kokkos::initialize( argc, argv );
  {

  // Allocate y, x vectors and Matrix A on device.
  typedef Kokkos::View<Particle*>   ViewVectorType;
  ViewVectorType p("p", N);

  // Create host mirrors of device views.
    ViewVectorType::HostMirror h_p = Kokkos::create_mirror_view( p ) ;

  Kokkos::parallel_for("init", N, KOKKOS_LAMBDA (int i){
	h_p(i).x = rand() / (double)RAND_MAX;
	h_p(i).y = rand() / (double)RAND_MAX;
	h_p(i).vx = 0.0 ;
	h_p(i).vy = 0.0 ; 
	h_p(i).mass = 1.0;
  });

  // Deep copy host views to device views.
    Kokkos::deep_copy(p, h_p);

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {

      Kokkos::parallel_for("kernel", N, KOKKOS_LAMBDA (int i){
      double fx = 0 , fy = 0 ;
      double dx , dy ; 
      double dist, dist3, force;
      for (int j = 0 ; j < N ; j++){
	dx = p(j).x - p(i).x ;
	dy = p(j).y - p(i).y ; 
	dist = sqrt(dx*dx + dy*dy);
	dist3 = dist * dist * dist;
	force = G * p(i).mass * p(j).mass / dist3 ; 
	fx += force * dx ; 
	fy += force * dy ;
      }
      // integration 
      p(i).vx += dt * fx / p(i).mass ; 
      p(i).vy += dt * fy / p(i).mass ;
      p(i).x += p(i).vy * dt ; 
      p(i).y += p(i).vy * dt ;
    });

  }


// now i purpose a hierarchical parallelism point of view
// we have nested loops (for i for j) but the integration is only for i 
// first method : we use parallel_reduce for them
// second method : we use a second array named fx and fy to store the intermediaire data and then launch a classic 1d parallel loop


  // Calculate time
  Kokkos::fence();
  double time = timer.seconds();
  std::cout << "Elapsed time : " << time << std::endl ; 
  }
  Kokkos::finalize();

  return 0;
}
