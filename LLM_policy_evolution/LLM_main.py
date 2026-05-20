from LLM_policy_evolution import evaluator


def main(dataset_name, seed):
    evaluator.main(dataset_name, seed)


if __name__ == '__main__':
    main('sN2h_1_5b2', 0)
