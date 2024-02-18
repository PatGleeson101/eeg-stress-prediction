column 1 is participant no. 1,2,3....,24
column 2 to 211 represents statistical features 
The suffix AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF refers to the 14 channels of the EEG headset

The extracted 15 features are the following:
	mean = average
	min = minimum
	max = maximum
	std = standard deviation
	var = variance
	iqr = interquartile range
	skw = skewness
	rms = root mean square
	sum = summation
	hjorth = hjorth parameter (mobility)
	hurst = hurst exponent
	mean_first_diff = mean of the first difference of the signals
	mean_second_diff = mean of the second difference of the signals
	apen = approxiamate entropy
	fuzzy = fuzzy entropy

For details on the features, please refer to this paper,

Rahman, J.S., Gedeon, T., Caldwell, S., Jones, R. and Jin, Z., 2021. Towards Effective Music Therapy for Mental Health Care Using Machine Learning Tools: Human Affective Reasoning and Music Genres. Journal of Artificial Intelligence and Soft Computing Research, 11(1), pp.5-20.

column 212 is the label
	1 = calm
	2 = stressful