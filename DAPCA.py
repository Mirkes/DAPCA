import numpy as np
import warnings


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols


def DAPCA(
    X,
    labels,
    nComp,
    Y=None,
    alpha=None,
    beta=0.9,
    gamma=0.4,
    kNN=1,
    kNNweights="uniform",
    delta=None,
    maxIter=5,
    tca=0,
    verbose="warning",
):
    """
    DAPCA calculated Domain Adaptation Principal Components (DAPCA) or,
    if matrix Y is empty, Supervised principal components (SPCA) for problem
    of classification. Detailed description of these types of principal
    components (PCom) can be found in 
      Gorban, A.N., Grechuk, B., Mirkes, E.M., Stasenko, S.V. and Tyukin,
      I.Y., 2021. High-dimensional separability for one-and few-shot
      learning. arXiv preprint arXiv:2106.15416. or in ???
    Usage
              [V, D, PX, PY] = DAPCA(X, Labels, Y, nComp)
    or
              [V, D, PX, PY] = DAPCA(X, Labels, Y, nComp, Name, Value)
    Inputs:
      X is n-by-d labelled data matrix. Each row of matrix contains one
          observation (case, record, object, instance).
      labels is n-by-1 vector of labels. Each unique value is considered as
          one class. Number of classes nClass is equal to number of unique
          values in array labels.
      Y is m-by-d unlabelled data matrix. Each row of matrix contains one
          observation (case, record, object, instance).
      nComp is number of required PComs and must be positive integer number
          less than d.
      Name, Value pairs can be one of the following type
          'alpha', nonnegative real number is attraction of projections of
              points with the same label in labelled dataset. 
              Default value is 0.
          'alpha', nClass-by-1 vector of nonnegative real numbers is vector
              of attraction of projections of points with the same label in
              labelled dataset. In this case each class have individual
              coefficient of attraction.
          'beta', positive real number is repulsion of projections of points
              of unlabelled dataset.
              Default value is 0.9.
          'gamma', positive real number is attraction between projection of
              unlabelled point and k nearest labelled neighbours.
              Default value is 0.4.
          'kNN', non-negative integer number is number of labelled nearest
              neighbours to use for attraction of each unlabbeled data point.
              Default value 1.
          'kNNweights', 'Uniform' required usage of the same weghts for all k
              neares neighbours.
              It is default value
          'kNNweights', 'dist' assumed usage of weights proportional to
              distances. This function is implemented as
                  function weights = distProp(distances)
                      weights = distances ./ sum(distnaces, 2)
                  
          'kNNweights', function handle is handle of function inputs and
              outputs described above.
          'delta', positive real number assumes usage the same number for all
              pairs of classes.
              Default value is 1.
          'delta', nClass-by-1 vector of positive real numbers R assumes
              usage of matrix Delta with elements Delta(i,j) = abs(R(i)-R(j))
              for i < j. 
          'delta', matrix with positive real numbers upper the main diagonal.
              Matrix must have size nClass-by-nClass. This method can be
              usefull for ordinal target attribute.
          'maxIter', positive integer number is maximal number of itterations
              for iterative DAPCA.
              Default value is 5.
          'TCA', positive real number is attraction coefficient for Transfer
              Component Analysis. TCA ignores DAPCA and usage of gamma.
          'verbose', 'none' means suppress all messages exclude raise ValueErrors,
              'warning' means communicate the raise ValueErrors and warnings, 'all'
              means add messages about number of used itteretions.
              Default value is 'Warning'.
    Outputs:
      V is d-by-nComp matrix with one PCom is each column.
      D is nComp-by-1 vector with the greatest nComp eigenvalues.
      PX is n-by-nComp matrix of projections of X onto nComp PComs.
      PY is m-by-nComp matrix of projections of Y onto nComp PComs.
      Data matrices X and Y MUST be complete: usage of NaN is forbidden. To
      impute missing values you can use any appropriate methods. For example,
      you can use kNNImpute or svdWithGaps from the Github repository
      https://github.com/Mirkes/DataImputation.
      Rows with NaN values will be removed.
    """

    # Sanity check of positional arguments
    # Type of X
    if ~np.isin(str(X.dtype), ['float64', 'float32', 'int32', 'int64']) or (len(X.shape) != 2):
        raise ValueError("the first argument must be numerical matrix")

    # Type of labels is arbitrary. Nothing to check. Convert to column
    # vector.
    labels = labels.reshape((-1, 1))

    # Remove NaNs from X
    ind = np.sum(np.isnan(X), 1)
    ind = ind > 0
    n_nans = sum(ind)
    if n_nans > 0:
        warnings.warn(
            f"Matrix X contains {n_nans} rows with missing values.",
            "All these rows were deleted",
        )
        X = X[~ind]
        labels = labels[~ind]

    # Calculate sizes of data
    nX, d = X.shape
    if nX == 0:
        raise ValueError(
            ["Matrix X must not be empty even after removing of", " incomplete rows"]
        )

    k = len(labels)
    if k != nX:
        raise ValueError(
            [
                "Number of elements in labels must be the same as number",
                " of row in matrix X",
            ]
        )

    labs, labNum = np.unique(labels, return_inverse=True)
    nClass = len(labs)
    # Type of matrix Y
    if Y is not None:
        ind = np.sum(np.isnan(Y), 1)
        ind = ind > 0
        if sum(ind) > 0:
            Y = Y[~ind]

    useY = True
    if Y is None:
        warnings.warn("Since matrix Y is empty the SPCA is used")
        useY = False
    else:
        nY, k = Y.shape
        if k != d:
            raise ValueError(
                "It is assumed that set of features in X and Y is the same"
            )

    # Value of nComp
    if (type(nComp) != int) or (np.floor(nComp) != nComp) or (nComp < 1):
        raise ValueError("nComp must be positive integer value")

    # Analyze values of optional arguments
    if alpha is None:
        alpha = np.zeros((nClass, 1))
    if delta is None:
        delta = np.ones((nClass,nClass))

    if type(alpha) == int:
        alpha = alpha * np.ones((nClass, 1))
    else:
        if len(alpha) != nClass:
            raise ValueError(
                'Value of parameter "alpha" must be either'
                + " nonnegative scalar or vector with number of classes"
                + f" {nClass} nonnegative elements"
            )

    if beta < 0:
        raise ValueError('Value of parameter "beta" must be positive scalar')

    if gamma < 0:
        raise ValueError('Value of parameter "beta" must be positive scalar')

    if (type(kNN) != int) or (kNN < 1):
        raise ValueError(
            [
                "kNN is number of labelled nearest neighbours and must",
                " be positive integer value",
            ]
        )

    if type(kNNweights) == str:
        if "uniform" == kNNweights:
            kNNweights = uniformWeights
        elif "dist" == kNNweights:
            kNNweights = distProp
        else:
            raise ValueError('Wrong value of argument "kNNweights"')

    # do we have raise ValueError?
    ind = False
    if (type(delta)==int) or (type(delta)==float) and (delta > 0):
        delta = delta * np.ones((nClass,nClass))

    elif (type(delta) == np.ndarray) and (len(delta.shape)==1) and np.all(delta > 0):
        # Form matrix
        tmp = np.zeros((nClass,nClass))
        for k in range(nClass):
            for kk in range(k + 1, nClass):
                tmp[k, kk] = np.abs(delta[k] - delta[kk])

        delta = tmp

    elif (
        (type(delta) == np.ndarray)
        and (len(delta.shape) == 2)
        and (len(delta) == nClass)
        and (delta.shape[1] == nClass)
    ):
        # now we need to check positivity of upper diagonal matrix
        tmp = np.sum(np.triu(delta, 1) > 0)
        if tmp != (nClass * (nClass - 1) / 2):
            ind = True


    else:
        ind = True

    if ind:
        raise ValueError(
            [
                "Argument delta must be positive real number, ",
                "vector with nClass positive elements, or matrix of size ",
                "nClass-by-nClass with positive real values upper the main diagonal",
            ]
        )

    if (np.floor(maxIter) != maxIter) or (maxIter < 1):
        raise ValueError(
            [
                "maxIter is maximal number of itterations for iterative",
                " DAPCA and must  be positive integer value",
            ]
        )

    if ~np.isin(type(tca), [int, float]) or (tca < 0):
        raise ValueError(
            [
                "Argument TCA must be positive real number which is",
                " attraction coefficient for Transfer Component Analysis.",
            ]
        )

    if tca > 0:
        if ~useY:
            raise ValueError("To use TCA it is necessary to specify Y matrix")

        warnings.warn("TCA is used and DAPCA parameters are ignored")

    # Check verbose
    tmp = False
    if type(verbose) == str:
        if "none" == verbose:
            verbose = 1
        elif "warning" == verbose:
            verbose = 2
        elif "all" == verbose:
            verbose = 3
        else:
            tmp = True

    else:
        tmp = True

    if tmp:
        raise ValueError(
            "Argument 'verbose' must be 'none' (means suppress"
            + " all messages exclude raise ValueErrors), 'warning' (means communicate"
            + " the raise ValueErrors and warnings), or 'all' (means add messages"
            + " about number of used iterations)."
        )


    # Reorder labels and X in order of labels
    ind = np.argsort(labels.flat,kind='mergesort')
    X = X[ind, :]
    # Calculate number of cases of each class n_i and means for classes
    # mu_i formula (6)?
    cnt = np.zeros((nClass, 1))
    means = np.zeros((nClass, d))
    for k in range(nClass):
        ind = labNum == k
        cnt[k] = np.sum(ind)
        means[k, :] = np.sum(X[ind, :], 0,keepdims=1)

    if useY:
        meanY = np.sum(Y, 0,keepdims=1)
        if tca > 0:
            meanX = np.mean(X, 0,keepdims=1)

    # Convert matrix delta to full matrix with -alpha on diagonal and delta
    # off diagonal and with normalisation by number of elements in each
    # class. Formulae (12)
    alpha = -alpha / (cnt * (cnt - 1))
    delta = delta / (cnt @ cnt.T)
    tmp = np.triu(delta, 1)
    delta = tmp + tmp.T + np.diag(alpha.flat)
    # Normalise other coefficients
    if useY:
        beta = beta / (nY * (nY - 1))
        gamma = gamma / (kNN * nY)

    # Calculation of constant parts of sum of weights vectors. Formulae
    # (13) and (10)
    tmp = delta @ cnt
    wX = np.repeat(tmp, cnt.astype(int).flat,axis=0)
    if useY:
        wY = np.tile(nY * beta, (nY, 1))

    # Calculation of constant parts of matrix Q.
    # Y part
    if useY:
        constQ = beta * (meanY.T @ meanY)
        if tca > 0:
            meanX = meanX - meanY / nY
            constQ = constQ + tca * (meanX.T @ meanX)

    else:
        constQ = np.zeros((d,d))

    # X Part
    for k in range(nClass):
        # diagonal part
        constQ = constQ + delta[k, k] * (means[[k], :].T @ means[[k], :])
        # Off diagonal part
        for kk in range(k + 1, nClass):
            tmp = delta[k, kk] * (means[[k], :].T @ means[[kk], :])
            constQ = constQ + tmp + tmp.T

    # Now we are ready for iterations.
    if useY and (tca == 0):
        kNNs = np.zeros((nY, kNN))
        kNNDist = kNNs.copy()
        # estimate step of Y records to calculate distances to all X records
        maxY = np.floor(1e8 / nX)
        if maxY > nY:
            maxY = nY

        maxY = maxY - 1
        PY = Y

    PX = X
    # Start iterations
    iterNum = 0
    while True:
        wXX = wX.copy()
        if useY:
            wYY = wY.copy()

        Q2 = constQ.copy()
        if useY and (tca == 0):
            # Remember old kNNs
            oldkNN = kNNs.copy()
            # Calculate new kNNs
            # calculate squared len of X vectors
            PX2 = np.sum(PX ** 2, 1,keepdims=1).T
            k = 1
            while k <= nY:
                # Define  of fragment
                kk = k + maxY
                if kk > nY:
                    kk = nY

                # Calculate distances
                dist = np.sum(PY[k - 1 : kk, :] ** 2, 1,keepdims=1) + PX2 - 2 * PY[k - 1 :kk, :] @ PX.T
                # Search NN
                ind = np.argsort(dist, 1, kind='mergesort')
                dist = dist[np.arange(len(dist))[:,None],ind]
                # Get kNN element
                kNNDist[k - 1 : kk, :] = -gamma * kNNweights(dist[:, :kNN])
                kNNs[k - 1 : kk, :] = ind[:, :kNN]
                # Correct wYY
                wYY[k - 1 : kk] = wYY[k - 1 : kk] + np.sum(kNNDist[k - 1 : kk, :], 1,keepdims=1)
                # Add summand to Q2
                nS = kk - k + 1
                tmp = np.zeros((nS, nX))
                #tmp[
                #    sub2ind(
                #        tmp.shape,
                #        np.tile(np.arange(nS)[:, None], (1, kNN)),
                #        ind[:, :kNN],
                #    )
                tmp[np.tile(np.arange(nS)[:, None], (1, kNN)),ind[:, :kNN]] = kNNDist[k - 1 :kk, :]
                tmp = Y[k - 1 : kk, :].T @ tmp @ X
                Q2 = Q2 + tmp + tmp.T
                # Shift k in Y
                k = kk + 1

            wXX = wXX + np.sum(kNNDist, 0,keepdims=1)
            if np.all(oldkNN == kNNs):
                break

        # X part of Q1
        Q1 = X.T @ (wXX * X)
        # Y part of Q1
        if useY:
            Q1 = Q1 + Y.T @ (wYY * Y)

        # Full matrix is
        Q = Q1 - Q2
        # Calculate principal components.
        D, V = np.linalg.eig(Q)
        # Sort eigenvalues
        ind = np.argsort(D,axis=0,kind='mergesort')[::-1]
        D = D[ind]
        V = V[:, ind]
        # Save the first nComp elements only
        D = D[:nComp]
        V = V[:, :nComp]
        # Standardise direction
        ind = np.sum(V, 0) < 0
        V[:, ind] = -V[:, ind]
        # Calculate projection
        PX = X @ V
        if useY:
            PY = Y @ V
        else:
            PY = np.array([])

        iterNum = iterNum + 1
        if (iterNum == maxIter) or not(useY) or (tca > 0):
            break

    if verbose > 2:
        print(f"Number of iterations {iterNum}\n")

    return V, D, PX, PY


def uniformWeights(distances):
    """ Calculate constant weights for all neighbours."""
    weights = np.ones_like(distances)
    return weights


def distProp(distances):
    """ Calculate weights proportional to relative distance."""
    weights = distances / np.max(distances, 1)
    return weights

