To compile a selected file write :
cmake -S. -B<a_directory_name> -DDTYPE=float

or


cmake -S. -B<a_directory_name> -DDTYPE=double

then

cmake --build <a_directory_name>

Results:
for float: -0.213894
for double: -6.76916e-10
