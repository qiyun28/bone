# bone

How to Use
----

1. save all the joints to a csv file *"joints.csv"*
    ```sh
    python joints_to_csv.py
    ```
    - Auto-detected error images will be printed at terminal. No information of these joints are saved to csv.
2. go through *"./for_checking"* folder to run another round of manual check.
3. add missing information (error from step 1) and/or correct false information (error from step 2) from the csv
    -  if you wish to check whether the edited information is correct run the following command for checking. A file named *"file_name.png"* with joint dots will be saved
        ```sh
        python check_csv.py <file_name,thumb1x,thumb1y,thumb2x,thumb2y,index1x,index1y,index2x,index2y,index3x,index3y,...,little3x,little3y>
        ```
        e.g.:
        ```sh
        python check_csv.py 2115,449,287,409,348,390,110,371,161,339,254,313,68,298,133,280,246,233,97,231,161,228,267,155,173,157,213,182,290
        ```
4. read from csv file *"joints.csv"* and save all joints coordinates to picture segments at *"./joints"*
    ```sh
    python csv_to_pic.py
    ```