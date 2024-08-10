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
  typedef Kokkos::View<double*>   ViewVectorType;
  //ViewVectorType y( "y", N );
  //ViewVectorType x( "x", M );
  //ViewMatrixType A( "A", N, M );
  ViewVectorType x("x", N);
  ViewVectorType y("y", N);  
  ViewVectorType vx("vx", N);
  ViewVectorType vy("vy", N);
  ViewVectorType mass("mass", N);  


  // Create host mirrors of device views.
//  ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y );
//  ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );
//  ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view( A );
    ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x ) ;
    ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y ) ;
    ViewVectorType::HostMirror h_vx = Kokkos::create_mirror_view( vx ) ;
    ViewVectorType::HostMirror h_vy = Kokkos::create_mirror_view( vy ) ;
    ViewVectorType::HostMirror h_mass = Kokkos::create_mirror_view( mass ) ;




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
		x(i) = rand() / (double)RAND_MAX;	
		y(i) = rand() / (double)RAND_MAX;		
		vx(i) = 0.0 ;
		vy(i) = 0.0 ; 
		mass(i) = 1.0;
  });

  // Deep copy host views to device views.
//  Kokkos::deep_copy( y, h_y );
//  Kokkos::deep_copy( x, h_x );
//  Kokkos::deep_copy( A, h_A );
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);
    Kokkos::deep_copy(vx, h_vx);
    Kokkos::deep_copy(vy, h_vy);
    Kokkos::deep_copy(mass, h_mass);


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
	dx = x(j) - x(i) ;
	dy = y(j) - y(i) ; 
	dist = sqrt(dx*dx + dy*dy);
	dist3 = dist * dist * dist;
	force = G * mass(i) * mass(j) / dist3 ; 
	fx += force * dx ; 
	fy += force * dy ;
      }
      // integration 
      vx(i) += dt * fx / mass(i) ; 
      vy(i) += dt * fy / mass(i) ;
      x(i) += vy(i) * dt ; 
      y(i) += vy(i) * dt ;
    });

  }

  // Calculate time.
  Kokkos::fence();
  double time = timer.seconds();
  std::cout << "Elapsed time : " << time << std::endl ; 
  }
  Kokkos::finalize();

  return 0;
}
