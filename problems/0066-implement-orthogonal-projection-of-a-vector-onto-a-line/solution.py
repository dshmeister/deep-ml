
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	# vL = v[0] * L[0] + v[1] * L[1]
	# LL = L[0] * L[0] + L[1] * L[1]
	# proj = [0] * 2
	# proj[0] = int((vL / LL) * L[0])
	# proj[1] = int((vL / LL) * L[1])

	# return proj
	vl = sum(v_i * L_i for v_i, L_i in zip(v, L))
	LL = sum(l_i * l_i for l_i in L)
	if LL == 0:
		return None
	coef = vl / LL
	return [coef * l_i for l_i in L]

