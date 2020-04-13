
def print_debugs(solver, verbose=True):
    print('Num. Iterations')
    print(solver.iter)
    print('Clock time')
    print(solver.time)
    if verbose:
        print('Policy')
        print(solver.policy)