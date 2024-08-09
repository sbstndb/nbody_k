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
  //ViewVectorType y( "y", N );
  //ViewVectorType x( "x", M );
  //ViewMatrixType A( "A", N, M );
  ViewVectorType p("p", N);

  // Create host mirrors of device views.
//  ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y );
//  ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );
//  ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view( A );
    ViewVectorType::HostMirror h_p = Kokkos::create_mirror_view( p ) ;

  // Initialize y vector on host.
//  for ( int i = 0; i < N; ++i ) {
//    h_y( i ) = 1;
//  }

  // Initialize x vector on host.
//  for ( int i = 0; i < M; ++i ) {
//    h_x( i ) = 1;
//  }

  // Initialize A matrix on host.
//  for ( int j = 0; j < N; ++j ) {
//    for ( int i = 0; i < M; ++i ) {
//      h_A( j, i ) = 1;
//    }
//  }
//
  Kokkos::parallel_for("init", N, KOKKOS_LAMBDA (int i){
	h_p(i).x = rand() / (double)RAND_MAX;
	h_p(i).y = rand() / (double)RAND_MAX;
	h_p(i).vx = 0.0 ;
	h_p(i).vy = 0.0 ; 
	h_p(i).mass = 1.0;
  });

  // Deep copy host views to device views.
//  Kokkos::deep_copy( y, h_y );
//  Kokkos::deep_copy( x, h_x );
//  Kokkos::deep_copy( A, h_A );
    Kokkos::deep_copy(p, h_p);

  // Timer products.
  Kokkos::Timer timer;

  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    // Application: <y,Ax> = y^T*A*x
//  double result = 0;

//    Kokkos::parallel_reduce( "yAx", N, KOKKOS_LAMBDA ( int j, double &update ) {
      Kokkos::parallel_for("kernel", N, KOKKOS_LAMBDA (int i){
//    double temp2 = 0;
//      for ( int i = 0; i < M; ++i ) {
//        temp2 += A( j, i ) * x( i );
//      }
//      update += y( j ) * temp2;
//    }, result );

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

  // Calculate time.
  double time = timer.seconds();
  std::cout << "Elapsed time : " << time << std::endl ; 
  }
  Kokkos::finalize();

  return 0;
}
