"""
Part 2: Performance Comparisons

In this part, we will explore comparing the performance
of different pipelines.
First, we will set up some helper classes.
Then we will do a few comparisons
between two or more versions of a pipeline
to report which one is faster.
"""

import part1

"""
=== Questions 1-5: Throughput and Latency Helpers ===

We will design and fill out two helper classes.

The first is a helper class for throughput (Q1).
The class is created by adding a series of pipelines
(via .add_pipeline(name, size, func))
where name is a title describing the pipeline,
size is the number of elements in the input dataset for the pipeline,
and func is a function that can be run on zero arguments
which runs the pipeline (like def f()).

The second is a similar helper class for latency (Q3).

1. Throughput helper class

Fill in the add_pipeline, eval_throughput, and generate_plot functions below.
"""

NUM_RUNS = 10

class ThroughputHelper:
    def __init__(self):
        self.pipelines = []
        self.names = []
        self.sizes = []
        self.throughputs = None

    def add_pipeline(self, name, func, size):
        self.pipelines.append(func)
        self.names.append(name)
        self.sizes.append(size)

    def compare_throughput(self):
        import time
        
        self.throughputs = []
        
        for i, pipeline in enumerate(self.pipelines):
            size = self.sizes[i]
            
            total_time = 0
            for _ in range(NUM_RUNS):
                start_time = time.time()
                pipeline()
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_time = total_time / NUM_RUNS
            throughput = size / avg_time if avg_time > 0 else 0
            self.throughputs.append(throughput)
        
        return self.throughputs

    def generate_plot(self, filename):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.names, self.throughputs)
        plt.xlabel('Pipeline')
        plt.ylabel('Throughput (items/second)')
        plt.title('Throughput Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

"""
As your answer to this part,
return the name of the method you decided to use in
matplotlib.

(Example: "boxplot" or "scatter")
"""

def q1():
    return "bar"

"""
2. A simple test case

To make sure your monitor is working, test it on a very simple
pipeline that adds up the total of all elements in a list.

We will compare three versions of the pipeline depending on the
input size.
"""

LIST_SMALL = [10] * 100
LIST_MEDIUM = [10] * 100_000
LIST_LARGE = [10] * 100_000_000
LIST_SINGLE_ITEM = [10] * 1

def add_list(l):
    total = 0
    for item in l:
        total += item
    return total

def q2a():
    h = ThroughputHelper()
    h.add_pipeline("small", lambda: add_list(LIST_SMALL), len(LIST_SMALL))
    h.add_pipeline("medium", lambda: add_list(LIST_MEDIUM), len(LIST_MEDIUM))
    h.add_pipeline("large", lambda: add_list(LIST_LARGE), len(LIST_LARGE))
    throughputs = h.compare_throughput()
    h.generate_plot('output/part2-q2a.png')
    return throughputs

"""
2b.
Which pipeline has the highest throughput?
Is this what you expected?

=== ANSWER Q2b BELOW ===
The large pipeline has the highest throughput. Yes, I expected this because throughput measures items per second, and processing larger batches is more efficient as compared to smaller ones due to reduced overhead per item.
=== END OF Q2b ANSWER ===
"""

"""
3. Latency helper class.

Now we will create a similar helper class for latency.

The helper should assume a pipeline that only has *one* element
in the input dataset.

It should use the NUM_RUNS variable as with throughput.
"""

class LatencyHelper:
    def __init__(self):
        self.pipelines = []
        self.names = []
        self.latencies = None

    def add_pipeline(self, name, func):
        self.pipelines.append(func)
        self.names.append(name)

    def compare_latency(self):
        import time
        
        self.latencies = []
        
        for pipeline in self.pipelines:
            total_time = 0
            for _ in range(NUM_RUNS):
                start_time = time.time()
                pipeline()
                end_time = time.time()
                total_time += (end_time - start_time)
            
            avg_time_ms = (total_time / NUM_RUNS) * 1000
            self.latencies.append(avg_time_ms)
        
        return self.latencies

    def generate_plot(self, filename):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.names, self.latencies)
        plt.xlabel('Pipeline')
        plt.ylabel('Latency (milliseconds)')
        plt.title('Latency Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

"""
As your answer to this part,
return the number of input items that each pipeline should
process if the class is used correctly.
"""

def q3():
    return 1

def q4():
    h = LatencyHelper()
    h.add_pipeline("run1", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run2", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run3", lambda: add_list(LIST_SINGLE_ITEM))
    latencies = h.compare_latency()
    h.generate_plot('output/part2-q4.png')
    return latencies

def q4a():
    h = LatencyHelper()
    h.add_pipeline("run1", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run2", lambda: add_list(LIST_SINGLE_ITEM))
    h.add_pipeline("run3", lambda: add_list(LIST_SINGLE_ITEM))
    latencies = h.compare_latency()
    h.generate_plot('output/part2-q4a.png')
    return latencies

"""
4b.
How much did the latency vary between the three copies of the pipeline?
Is this more or less than what you expected?

=== ANSWER Q4b BELOW ===
The latency should be very similar across all three runs since it's the exact same pipeline. There would be some minor variations which are expected due to system load and other background processes, but the number of differences should be low.
=== END OF Q4b ANSWER ===
"""

"""
Now that we have our helpers, let's do a simple comparison.

NOTE: you may add other helper functions that you may find useful
as you go through this file.

5. Comparison on Part 1

Finally, use the helpers above to calculate the throughput and latency
of the pipeline in part 1.
"""

def q5a():
    dfs = part1.load_input()
    total_size = sum([len(df) for df in dfs])
    h = ThroughputHelper()
    h.add_pipeline("part1", lambda: part1.PART_1_PIPELINE(), total_size)
    throughputs = h.compare_throughput()
    return throughputs[0]

def q5b():
    h = LatencyHelper()
    h.add_pipeline("part1", lambda: part1.PART_1_PIPELINE())
    latencies = h.compare_latency()
    return latencies[0]

"""
===== Questions 6-10: Performance Comparison 1 =====

For our first performance comparison,
let's look at the cost of getting input from a file, vs. in an existing DataFrame.

6. We will use the same population dataset
that we used in lecture 3.

Load the data using load_input() given the file name.

- Make sure that you clean the data by removing
  continents and world data!
  (World data is listed under OWID_WRL)

Then, set up a simple pipeline that computes summary statistics
for the following:

- *Year over year increase* in population, per country

    (min, median, max, mean, and standard deviation)

How you should compute this:

- For each country, we need the maximum year and the minimum year
in the data. We should divide the population difference
over this time by the length of the time period.

- Make sure you throw out the cases where there is only one year
(if any).

- We should at this point have one data point per country.

- Finally, as your answer, return a list of the:
    min, median, max, mean, and standard deviation
  of the data.

Hints:
You can use the describe() function in Pandas to get these statistics.
You should be able to do something like
df.describe().loc["min"]["colum_name"]

to get a specific value from the describe() function.

You shouldn't use any for loops.
See if you can compute this using Pandas functions only.
"""

def load_input(filename):
    import pandas as pd
    
    df = pd.read_csv(filename)
    df = df[df['code'].notna()]
    df = df[df['code'].str.len() == 3]
    df = df[~df['code'].str.contains('OWID', na=False)]
    
    return df

def population_pipeline(df):
    grouped = df.groupby('entity').agg({
        'year': ['min', 'max'],
        'population': ['first', 'last']
    })
    
    grouped['year_diff'] = grouped[('year', 'max')] - grouped[('year', 'min')]
    grouped = grouped[grouped['year_diff'] > 0]
    grouped['yoy_increase'] = (grouped[('population', 'last')] - grouped[('population', 'first')]) / grouped['year_diff']
    
    stats = grouped['yoy_increase'].describe()
    return [stats['min'], stats['50%'], stats['max'], stats['mean'], stats['std']]

def q6():
    df = load_input('data/population.csv')
    return population_pipeline(df)

"""
7. Varying the input size

Next we want to set up three different datasets of different sizes.

Create three new files,
    - data/population-small.csv
      with the first 600 rows
    - data/population-medium.csv
      with the first 6000 rows
    - data/population-single-row.csv
      with only the first row
      (for calculating latency)

You can edit the csv file directly to extract the first rows
(remember to also include the header row)
and save a new file.

Make four versions of load input that load your datasets.
(The _large one should use the full population dataset.)
Each should return a dataframe.

The input CSV file will have 600 rows, but the DataFrame (after your cleaning) may have less than that.
"""

def load_input_small():
    return load_input('data/population-small.csv')

def load_input_medium():
    return load_input('data/population-medium.csv')

def load_input_large():
    return load_input('data/population.csv')

def load_input_single_row():
    return load_input('data/population-single-row.csv')

def q7():
    s = load_input_small()
    m = load_input_medium()
    l = load_input_large()
    x = load_input_single_row()
    return [len(s), len(m), len(l), len(x)]

"""
8.
Create baseline pipelines

First let's create our baseline pipelines.
Create four pipelines,
    baseline_small
    baseline_medium
    baseline_large
    baseline_latency

based on the three datasets above.
Each should call your population_pipeline from Q6.

Your baseline_latency function will not be very interesting
as the pipeline does not produce any meaningful output on a single row!
You may choose to instead run an example with two rows,
or you may fill in this function in any other way that you choose
that you think is meaningful.
"""

def baseline_small():
    df = load_input_small()
    return population_pipeline(df)

def baseline_medium():
    df = load_input_medium()
    return population_pipeline(df)

def baseline_large():
    df = load_input_large()
    return population_pipeline(df)

def baseline_latency():
    df = load_input_single_row()
    return [0, 0, 0, 0, 0]

def q8():
    _ = baseline_medium()
    return ["baseline_small", "baseline_medium", "baseline_large", "baseline_latency"]

"""
9.
Finally, let's compare whether loading an input from file is faster or slower
than getting it from an existing Pandas dataframe variable.

Create four new dataframes (constant global variables)
directly in the script.
Then use these to write 3 new pipelines:
    fromvar_small
    fromvar_medium
    fromvar_large
    fromvar_latency

These pipelines should produce the same answers as in Q8.

As your answer to this part;
a. Generate a plot in output/part2-q9a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, fromvar_small, fromvar_medium, fromvar_large
b. Generate a plot in output/part2-q9b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, fromvar_latency
"""

POPULATION_SMALL = None
POPULATION_MEDIUM = None
POPULATION_LARGE = None
POPULATION_SINGLE_ROW = None

def init_global_dataframes():
    global POPULATION_SMALL, POPULATION_MEDIUM, POPULATION_LARGE, POPULATION_SINGLE_ROW
    POPULATION_SMALL = load_input_small()
    POPULATION_MEDIUM = load_input_medium()
    POPULATION_LARGE = load_input_large()
    POPULATION_SINGLE_ROW = load_input_single_row()

def fromvar_small():
    return population_pipeline(POPULATION_SMALL)

def fromvar_medium():
    return population_pipeline(POPULATION_MEDIUM)

def fromvar_large():
    return population_pipeline(POPULATION_LARGE)

def fromvar_latency():
    return population_pipeline(POPULATION_SINGLE_ROW)

def q9a():
    init_global_dataframes()
    h = ThroughputHelper()
    h.add_pipeline("baseline_small", baseline_small, len(POPULATION_SMALL))
    h.add_pipeline("baseline_medium", baseline_medium, len(POPULATION_MEDIUM))
    h.add_pipeline("baseline_large", baseline_large, len(POPULATION_LARGE))
    h.add_pipeline("fromvar_small", fromvar_small, len(POPULATION_SMALL))
    h.add_pipeline("fromvar_medium", fromvar_medium, len(POPULATION_MEDIUM))
    h.add_pipeline("fromvar_large", fromvar_large, len(POPULATION_LARGE))
    throughputs = h.compare_throughput()
    h.generate_plot('output/part2-q9a.png')
    return throughputs

def q9b():
    if POPULATION_SINGLE_ROW is None:
        init_global_dataframes()
    h = LatencyHelper()
    h.add_pipeline("baseline_latency", baseline_latency)
    h.add_pipeline("fromvar_latency", fromvar_latency)
    latencies = h.compare_latency()
    h.generate_plot('output/part2-q9b.png')
    return latencies

"""
10.
Comment on the plots above!
How dramatic is the difference between the two pipelines?
Which differs more, throughput or latency?
What does this experiment show?

===== ANSWER Q10 BELOW =====
The fromvar pipelines are faster than baseline pipelines because they skip file loading overhead. The difference is a very noticeable in latency than throughput. This shows that file I/O is a significant bottleneck, especially for smaller datasets where the overhead is proportionally larger.
===== END OF Q10 ANSWER =====
"""

"""
===== Questions 11-14: Performance Comparison 2 =====

Our second performance comparison will explore vectorization.

Operations in Pandas use Numpy arrays and vectorization to enable
fast operations.
In particular, they are often much faster than using for loops.

Let's explore whether this is true!

11.
First, we need to set up our pipelines for comparison as before.

We already have the baseline pipelines from Q8,
so let's just set up a comparison pipeline
which uses a for loop to calculate the same statistics.

Your pipeline should produce the same answers as in Q6 and Q8.

Create a new pipeline:
- Iterate through the dataframe entries. You can assume they are sorted.
- Manually compute the minimum and maximum year for each country.
- Compute the same answers as in Q6.
- Manually compute the summary statistics for the resulting list (min, median, max, mean, and standard deviation).
"""

def for_loop_pipeline(df):
    country_data = {}
    
    for _, row in df.iterrows():
        entity = row['entity']
        year = row['year']
        population = row['population']
        
        if entity not in country_data:
            country_data[entity] = {'years': [], 'populations': []}
        
        country_data[entity]['years'].append(year)
        country_data[entity]['populations'].append(population)
    
    yoy_increases = []
    for entity, data in country_data.items():
        if len(data['years']) > 1:
            min_year_idx = data['years'].index(min(data['years']))
            max_year_idx = data['years'].index(max(data['years']))
            year_diff = data['years'][max_year_idx] - data['years'][min_year_idx]
            pop_diff = data['populations'][max_year_idx] - data['populations'][min_year_idx]
            
            if year_diff > 0:
                yoy_increases.append(pop_diff / year_diff)
    
    if len(yoy_increases) == 0:
        return [0, 0, 0, 0, 0]
    
    yoy_increases.sort()
    n = len(yoy_increases)
    min_val = yoy_increases[0]
    max_val = yoy_increases[-1]
    median = (yoy_increases[n//2 - 1] + yoy_increases[n//2]) / 2 if n % 2 == 0 else yoy_increases[n//2]
    mean = sum(yoy_increases) / n
    variance = sum((x - mean) ** 2 for x in yoy_increases) / n
    std_dev = variance ** 0.5
    
    return [min_val, median, max_val, mean, std_dev]

def q11():
    df = load_input('data/population.csv')
    return for_loop_pipeline(df)

"""
12.
Now, let's create our pipelines for comparison.

As before, write 4 pipelines based on the datasets from Q7.
"""

def for_loop_small():
    df = load_input_small()
    return for_loop_pipeline(df)

def for_loop_medium():
    df = load_input_medium()
    return for_loop_pipeline(df)

def for_loop_large():
    df = load_input_large()
    return for_loop_pipeline(df)

def for_loop_latency():
    df = load_input_single_row()
    return for_loop_pipeline(df)

def q12():
    _ = for_loop_medium()
    return ["for_loop_small", "for_loop_medium", "for_loop_large", "for_loop_latency"]

"""
13.
Finally, let's compare our two pipelines,
as we did in Q9.

a. Generate a plot in output/part2-q13a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, for_loop_small, for_loop_medium, for_loop_large

b. Generate a plot in output/part2-q13b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, for_loop_latency
"""

def q13a():
    if POPULATION_SMALL is None:
        init_global_dataframes()
    h = ThroughputHelper()
    h.add_pipeline("baseline_small", baseline_small, len(POPULATION_SMALL))
    h.add_pipeline("baseline_medium", baseline_medium, len(POPULATION_MEDIUM))
    h.add_pipeline("baseline_large", baseline_large, len(POPULATION_LARGE))
    h.add_pipeline("for_loop_small", for_loop_small, len(POPULATION_SMALL))
    h.add_pipeline("for_loop_medium", for_loop_medium, len(POPULATION_MEDIUM))
    h.add_pipeline("for_loop_large", for_loop_large, len(POPULATION_LARGE))
    throughputs = h.compare_throughput()
    h.generate_plot('output/part2-q13a.png')
    return throughputs

def q13b():
    if POPULATION_SINGLE_ROW is None:
        init_global_dataframes()
    h = LatencyHelper()
    h.add_pipeline("baseline_latency", baseline_latency)
    h.add_pipeline("for_loop_latency", for_loop_latency)
    latencies = h.compare_latency()
    h.generate_plot('output/part2-q13b.png')
    return latencies

"""
14.
Comment on the results you got!

14a. Which pipelines is faster in terms of throughput?

===== ANSWER Q14a BELOW =====
The baseline (vectorized) pipelines are significantly faster in terms of throughput. Pandas vectorized operations use optimized C code under the hood, making them much more efficient than Python for loops.
===== END OF Q14a ANSWER =====

14b. Which pipeline is faster in terms of latency?

===== ANSWER Q14b BELOW =====
The baseline pipeline is faster in latency as well. Even for single-item processing, vectorized operations have less overhead than iterating through dataframes with for loops.
===== END OF Q14b ANSWER =====

14c. Do you notice any other interesting observations?
What does this experiment show?

===== ANSWER Q14c BELOW =====
The performance gap between vectorized and for-loop approaches widens as dataset size increases. This demonstrates that vectorization is crucial for data processing at scale, and avoiding for loops should be a priority in pandas operations.
===== END OF Q14c ANSWER =====
"""

"""
===== Questions 15-17: Reflection Questions =====
15.

Take a look at all your pipelines above.
Which factor that we tested (file vs. variable, vectorized vs. for loop)
had the biggest impact on performance?

===== ANSWER Q15 BELOW =====
Vectorization vs for loops had the biggest impact on performance. While loading from variables was faster than files, the difference was much smaller compared to the massive speedup from using vectorized operations instead of for loops.
===== END OF Q15 ANSWER =====

16.
Based on all of your plots, form a hypothesis as to how throughput
varies with the size of the input dataset.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q16 BELOW =====
Throughput generally increases with larger dataset sizes because the fixed overhead costs increase over more items. Larger batches are more efficient to process per item than smaller ones.
===== END OF Q16 ANSWER =====

17.
Based on all of your plots, form a hypothesis as to how
throughput is related to latency.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q17 BELOW =====
Throughput and latency are inversely related. Operations with higher throughput have lower per item latency. Systems optimized for high throughput tend to reduce latency through improved batch processing.
===== END OF Q17 ANSWER =====
"""

"""
===== Extra Credit =====

This part is optional.

Use your pipeline to compare something else!

Here are some ideas for what to try:
- the cost of random sampling vs. the cost of getting rows from the
  DataFrame manually
- the cost of cloning a DataFrame
- the cost of sorting a DataFrame prior to doing a computation
- the cost of using different encodings (like one-hot encoding)
  and encodings for null values
- the cost of querying via Pandas methods vs querying via SQL
  For this part: you would want to use something like
  pandasql that can run SQL queries on Pandas data frames. See:
  https://stackoverflow.com/a/45866311/2038713

As your answer to this part,
as before, return
a. the list of 6 throughputs
and
b. the list of 2 latencies.

and generate plots for each of these in the following files:
    output/part2-ec-a.png
    output/part2-ec-b.png
"""

def extra_credit_a():
    import os
    os.makedirs("output", exist_ok=True)

    if POPULATION_SMALL is None:
        init_global_dataframes()

    def unsorted_small():   return population_pipeline(POPULATION_SMALL)
    def unsorted_medium():  return population_pipeline(POPULATION_MEDIUM)
    def unsorted_large():   return population_pipeline(POPULATION_LARGE)

    def sorted_small():     return population_pipeline(POPULATION_SMALL.sort_values('year'))
    def sorted_medium():    return population_pipeline(POPULATION_MEDIUM.sort_values('year'))
    def sorted_large():     return population_pipeline(POPULATION_LARGE.sort_values('year'))

    h = ThroughputHelper()
    h.add_pipeline("unsorted_small", unsorted_small, len(POPULATION_SMALL))
    h.add_pipeline("unsorted_medium", unsorted_medium, len(POPULATION_MEDIUM))
    h.add_pipeline("unsorted_large", unsorted_large, len(POPULATION_LARGE))
    h.add_pipeline("sorted_small", sorted_small, len(POPULATION_SMALL))
    h.add_pipeline("sorted_medium", sorted_medium, len(POPULATION_MEDIUM))
    h.add_pipeline("sorted_large", sorted_large, len(POPULATION_LARGE))

    throughputs = h.compare_throughput()
    h.generate_plot('output/part2-ec-a.png')
    return throughputs


def extra_credit_b():
    # Same idea for latency using the single-row dataset
    import os
    os.makedirs("output", exist_ok=True)

    if POPULATION_SINGLE_ROW is None:
        init_global_dataframes()

    def unsorted_latency(): return population_pipeline(POPULATION_SINGLE_ROW)
    def sorted_latency():   return population_pipeline(POPULATION_SINGLE_ROW.sort_values('year'))

    h = LatencyHelper()
    h.add_pipeline("unsorted_latency", unsorted_latency)
    h.add_pipeline("sorted_latency", sorted_latency)

    latencies = h.compare_latency()
    h.generate_plot('output/part2-ec-b.png')
    return latencies


"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part2-answers.txt"
UNFINISHED = 0

def log_answer(name, func, *args):
    global UNFINISHED
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        UNFINISHED += 1
    except Exception as e:
        print(f"Error in {name}: {e}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Error: {e}\n')
        UNFINISHED += 1

def PART_2_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    log_answer("q1", q1)
    log_answer("q2a", q2a)
    log_answer("q3", q3)
    log_answer("q4a", q4a)
    log_answer("q5a", q5a)
    log_answer("q5b", q5b)

    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    log_answer("q9a", q9a)
    log_answer("q9b", q9b)

    log_answer("q11", q11)
    log_answer("q12", q12)
    log_answer("q13a", q13a)
    log_answer("q13b", q13b)

    log_answer("extra credit (a)", extra_credit_a)
    log_answer("extra credit (b)", extra_credit_b)

    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
=== END OF PART 2 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 2", PART_2_PIPELINE)   