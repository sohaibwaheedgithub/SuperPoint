class Evaluator:
    def evaluate(self, model, dataset) -> dict:
        """
        Runs validation and returns a flat metrics dict.
        """
        raise NotImplementedError
