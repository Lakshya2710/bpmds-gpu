#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include <vector>


// Useful constants
constexpr double PI = 3.14159265358979323846;
const double EPS = 1e-3;  // small tolerance for floating-point comparison

// Data types
using cord_t      = double; // Type for x and y co-ordinates
using distance_t  = double; // Type for distance between two vertices 
using demand_t    = double; // Type for demand of a customer
using capacity_t  = double; // Type for capacity of the vehicles
using node_t      = int; // Type for unique id for a node

// Useful things for error handling
extern std::ostream& ERROR_FILE;
void handle_error(const char*, int, std::ostream& out, const std::string, const bool);
#define HANDLE_ERROR(msg, exit_flag) handle_error(__FILE__, __LINE__, ERROR_FILE, msg, exit_flag)

class Point
{
    /*
    * 2-D cordinates & demand of a customer (or) depot
    */    
public:
    cord_t x;
    cord_t y;
    demand_t demand;

    Point();
    Point(cord_t, cord_t, demand_t);
};

double get_curr_rss_mb();
double get_peak_rss_mb();

#endif
