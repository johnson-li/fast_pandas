Group and Transfer
===

We provide a custom API to provide equivalent functionality of pandas' group and transfer. Please refer the [benchmark code](../quick_pandas/benchmark/groupby.py) for how to use it.

Generally, our version of group and transfer runs faster than pandas and supports custom definition of the transfer functions. Since the backend logic is implemented by Cython, calling custom function at the speed of c is nontrivial.

A few setups are required. For example, we want to transform all of the values inside a group to the maximum.

1. Clone the code from git
2. Create a pyx file inside [quick_pandas/ext](../quick_pandas/ext), namely demo_max.pyx. Note that you are free to put demo_max.pyx anywhere as long that you know how to compile it. To be convenient, you can utilize [setup.py](../setup.py) to compile all pyx file automatically.
3. We have created the file and filled in the content for you. Please refer the code for how it works. Basically, you need to a) define 4 function for maximum to handle 4 different data types separately and b) write your code to invoke our group and transform API.
4. Compile the code with Cython.
    ```bash
    python setup.py build_ext --inplace
    ```
5. Run your code.
    ```python
    import quick_pandas.ext.demo_max
    max.test()
    ```
6. There is a simple benchmark result that shows how our implementation out-performs pandas.
    ```python
    import quick_pandas.ext.demo_max
    max.benchmark()
    ```
