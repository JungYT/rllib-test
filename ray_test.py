import numpy as np
import time
import ray
from ray.exceptions import GetTimeoutError, TaskCancelledError
from ray import tune

import fym

@ray.remote
def f(x):
    return x*x

## returns multiple object refs
@ray.remote(num_returns=3)
def fs():
    return 1, 2, 3

# @ray.remote
# def blocking():
#     time.sleep(10e6)

@ray.remote
class Counter():
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

def objective(step, alpha, beta):
    return (0.1 + alpha*step/100)**(-1) + beta*0.1

def training_function(config):
    alpha, beta = config['alpha'], config['beta']
    for step in range(10):
        intermediate_score = objective(step, alpha, beta)
        tune.report(mean_loss=intermediate_score)

def main():
    ## Multiple oject refs in list
    fus = fs.remote()
    a, b, c = fs.remote()

    ## Parallel compute
    futures = [f.remote(i) for i in range(4)]

    ## Passing object refs to remote functions
    futures2 = [f.remote(i) for i in fus]

    ## Objects in Ray. It is similiar to pointer?
    y = np.array([1, 2])
    object_ref = ray.put(y)

    ## Actors
    counter_actor = Counter.remote()
    counter_obj = counter_actor.increment.remote()

    ## Print
    print(ray.get(futures))
    print(ray.get(a))
    print(ray.get(b))
    print(ray.get(c))
    [print(ray.get(fus[i])) for i in range(3)]
    print(ray.get(futures2))
    print(y)
    print(ray.get(object_ref))
    print(ray.get(counter_obj)) # This prints 1 because function is stateless
    print(ray.get(counter_obj)) # This prints 1
    print(ray.get(counter_obj)) # This prints 1
    print(ray.get(counter_actor.increment.remote())) # This prints 2 because
    ## class is stateful
    print(ray.get(counter_actor.increment.remote())) # This prints 3

    ## Tune
    analysis = tune.run(
        training_function,
        config={
            'alpha': tune.grid_search([0.001, 0.01, 0.1]),
            'beta': tune.choice([1, 2, 3])
        }
    )

    print("best config: ", analysis.get_best_config(
        metric="mean_loss",
        mode="min"
    ))

    df = analysis.results_df

    fym.config.update({"ray":1})





if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()

