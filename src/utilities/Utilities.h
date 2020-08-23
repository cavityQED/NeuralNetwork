#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <vector>

#include <cstdlib>
#include <ctime>
#include <chrono>

class Utilities
{
private:
	static int ID;
	using hiResClock = std::chrono::high_resolution_clock;
	using millisecond = std::chrono::duration<double, std::ratio< 1, 1000 > >;
	using second = std::chrono::duration<double, std::ratio<1, 1> > ;
	using minute = std::chrono::duration<double, std::ratio<60, 1> >;
	using timePoint = std::chrono::time_point<hiResClock, millisecond>;
	using duration = std::chrono::duration<double>;

	timePoint start;
	timePoint masterStartTime;
	timePoint end;
	timePoint minTimePoint = start.min();

	template <typename T>
	void swap(T* a, T* b)
	{
		T t = *a;
		*a = *b;
		*b = t;
	}

	template <typename T>
	static bool ascending(T a, T b) { return a<b; }

	template <typename T>
	void partition(std::vector<T> &vec, int low, int high, int &lowPivot, int &highPivot, bool (*func)(T, T))
	{
		T pivot = ((vec[high]+vec[low])/2);
		int i = low - 1;
		int k = 0;
		int end = high;

		for(int j=low; j<=high; j++)
		{
			if(func(vec[j], pivot))
			{
				i++;
				swap(&vec[i], &vec[j]);
			}
			else if(vec[j] == pivot)
			{
				swap(&vec[j], &vec[high]);
				k++;
				j--;
				high--;
			}
		}
		for(int x = 0; x < k; x++)
		{
			swap(&vec[end-x], &vec[i+x+1]);
		}
		lowPivot = i;
		highPivot = i+k+1;
	}

public:
	//Constructor 
	//		-Initializes std::rand with time clock
	Utilities();

	//Deconstructor
	~Utilities();

	int getID() { ID++; return ID; }

	//Funtion to return a random integer between min and max
	int RNG( int min, int max );

	template <typename T>
	T RNG( T min, T max )
	{
		static constexpr double fraction { 1.0 / ( 1.0 + RAND_MAX ) };
		return static_cast<T>( min + ( (max - min) * ( std::rand() * fraction ) ) );
	}

	void startTimer();

	double getElapsedTime();
	void printElapsedTime();

	void wait( int ms );

	static double strToDouble( const std::string &str );

	template <typename T>
	void sort(std::vector<T> &vec, int low, int high, bool (*func)(T, T) = ascending)
	{
		int lowPivot;
		int highPivot;

		if(low < high)
		{
			partition(vec, low, high, lowPivot, highPivot, func);
			sort(vec, low, lowPivot, func);
			sort(vec, highPivot, high, func);
		}
	}

	template <typename T>
	void printVector(std::vector<T> &vec)
	{
		auto it = vec.begin();
		while(it != vec.end())
		{
			std::cout << *it << ", ";
			it++;
		}
		std::cout << '\n';
	}

	template <typename T>
	bool isOrdered(std::vector<T> &vec)
	{
		for(int i = 0; i < vec.size()-1; i++)
		{
			if(vec[i] > vec[i+1])
				return false;
		}
		return true;
	}
};

#endif