import tensorflow as tf

@tf.function
def predict_local(model, args):
	return model(args)