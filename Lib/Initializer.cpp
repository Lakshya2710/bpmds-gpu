#include "Command_Line_Args.h"
#include <iomanip>

void initialize(const Command_Line_Args& args)
{
    args.output() << std::fixed << std::setprecision(4);
}