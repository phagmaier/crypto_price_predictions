import numpy as np

def percent_to_val(preds,targets):
	price_preds = [preds[0]/100 * targets[0] + targets[0]] + [(preds[i]/100 * targets[i-1] + targets[i-1]) for i in range(1,len(preds))]
	targets = targets[1:]
	errors = sum([(abs(i-x)) for i,x in zip(price_preds,targets)]) / len(price_preds)
	return price_preds, errors

def price_error(preds,targets):
	return sum(abs(i-x) for i,x in zip(preds,targets))/len(targets)


def get_ensemble_pred(preds, errors):
    weights = [1/i for i in errors]
    total_weight = sum(weights)
    normalized_weights = [weight/total_weight for weight in weights]
    def helper(preds, weight):
        return [i * weight for i in preds]

    weighted_preds = zip(*[helper(i, x) for i, x in zip(preds, normalized_weights)])
    return [sum(elements) for elements in weighted_preds]














