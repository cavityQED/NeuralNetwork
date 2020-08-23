#include "Utilities.h"

#include <iostream>
#include <cstdlib>
#include <ratio>
#include <chrono>
#include <thread>
#include <cassert>
#include <cmath>

int Utilities::ID = 0;
Utilities::Utilities()
{
	//start the clock for random number generation use
	std::srand( static_cast<unsigned int>( time(0) ) );
	start = minTimePoint;
	masterStartTime = hiResClock::now();

}

Utilities::~Utilities()
{
	std::chrono::duration<double> span = hiResClock::now() - masterStartTime;
	std::cout << "Total time: " << std::chrono::duration_cast<minute>(span).count() << " minutes\n";
}

int Utilities::RNG( int min, int max )
{
	static constexpr double fraction { 1.0 / ( 1.0 + RAND_MAX ) };
	return static_cast<int>( min + ( (max - min + 1) * ( std::rand() * fraction ) ) );
}

void Utilities::startTimer()
{
	start = hiResClock::now();
}

double Utilities::getElapsedTime()
{
	assert( start != minTimePoint && "Timer has not been started" );
	end = hiResClock::now();
	std::chrono::duration<double> span = end - start;

	return std::chrono::duration_cast<millisecond>(span).count();

}

void Utilities::printElapsedTime()
{
	assert( start != minTimePoint && "Timer has not been started" );
	end = hiResClock::now();
	std::chrono::duration<double> span = end - start;

	std::cout << std::chrono::duration_cast<millisecond>(span).count() << "ms have passed\n";
}

void Utilities::wait( int ms )
{
	std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );
}

double Utilities::strToDouble( const std::string &str )
{
	double lhs;
	double rhs;

	double temp;

	double rhs_count = 0;

	auto it = str.begin();


	while( *it != '.' )
	{
		temp = static_cast<double>(*it)-48.0;
		lhs = lhs*10 + temp;
		it++;
	}
	it++;
	while( it != str.end() )
	{
		temp = static_cast<double>(*it) - 48.0;
		rhs = rhs*10 + temp;
		rhs_count++;
		it++;
	}

	rhs = rhs/pow(10, rhs_count);

	std::cout << "lhs: " << lhs << '\n';
	std::cout << "rhs: " << rhs << '\n';

	return lhs+rhs;
}

