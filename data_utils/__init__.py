#__init__.py

"""
We rework the inputs and outputs of the model in several ways:

	We allow a transaction id to be passed to the categorisation function and return with the results
	We restructure the results to be a list of parameters for each transaction instead of lists by parameter type
	We add group metrics as an output at the transaction-level e.g. groupid, keywords, amounts, freq, etc
	We include the cut-off as an input parameter to reclassify low-scoring transactions
	We enhance the "show me the categories function" to display this extra information to the console

The corresponding I&E model is also updated to allow transaction-level reporting of I&E category and passing through of groupids and properties at the same level

"""


