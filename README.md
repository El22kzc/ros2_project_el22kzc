# ros2-project
The robot will move and explore the map using coordinates, to detect green,red and blue boxes. The robot will move up to the blue box and stop at roughly within a radius of 1 meter from the centre of the blue box after detecting all 3 colours.If it hadnt detected 3 colour, it will continue exploring the map until they are all detected.Also if it detected blue first, it will remember will it detected blue and continue exploring the map until all colour are detected once they are detected, it
will go back to the point where it saw blue first then find blue and will move up to the blue box and stop at roughly within a radius of 1 meter from the centre of the blue box.

The source code is ros2_project_el22kzc/robot.py
