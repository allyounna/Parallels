To compile a selected file write :
cmake -S. -B<a directory name> -DDTYPE=float
or
cmake -S. -B<a directory name> -DDTYPE=double
then
cmake --build <a directory name>

Results:
for float: -0.213894
for double: -6.76916e-10
