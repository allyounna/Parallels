To compile a selected file write :

  1.
    for float: cmake -S. -B<a_directory_name> -DDTYPE=float
    for double: cmake -S. -B<a_directory_name> -DDTYPE=double
    
  2.
    cmake --build <a_directory_name>

To launch a programm go to the <a_directory_name> directory and write :

    ./sum

Results:

  for float: -0.213894
  
  for double: -6.76916e-10
