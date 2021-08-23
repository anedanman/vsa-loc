import numpy as np

global VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION

VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION = 1000


def set_dimension(vsa_dimension=1000):
    """Set VSA dimension."""
    
    global VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION
    VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION = vsa_dimension
    

def get_dimension():
    """Print VSA dim."""
    
    return globals()['VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION']


def generate(r_state=None):
    """Generate a random real hypervector of dimension dim.

    Default dimension dim -- 1000
    """

    dim = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION']

    if r_state is None:
        r_state = np.random.RandomState()
    v = r_state.randn(dim)
    v /= np.linalg.norm(v)
    return v


def make_unitary(v):
    fft_val = np.fft.fft(v)
    fft_imag = fft_val.imag
    fft_real = fft_val.real
    fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
    invalid = fft_norms <= 0.0
    fft_val[invalid] = 1.0
    fft_norms[invalid] = 1.0
    fft_unit = fft_val / fft_norms
    return np.array((np.fft.ifft(fft_unit, n=len(v))).real)


def power(v, e):
    x = np.fft.ifft(np.fft.fft(v) ** e).real
    return x


def bind(hd_vec_1, hd_vec_2):
    n = len(hd_vec_1)
    return np.fft.irfft(np.fft.rfft(hd_vec_1) * np.fft.rfft(hd_vec_2), n=n)


def bundle(vec1, vec2):
    result = vec1 + vec2
    return result / np.linalg.norm(result)


def bundle2(*hd_vectors):
    hd_vectors = np.array(hd_vectors).squeeze(0)  # Get rid of a dimension introduced by a star operator
    result = np.array(hd_vectors).sum(axis=0)
    return result / np.linalg.norm(result)


def sim(v1, v2):
    scale = np.linalg.norm(v1) * np.linalg.norm(v2)
    if scale == 0:
        return 0
    return np.dot(v1, v2) / scale


class ItemMemory:
    """Store noiseless hypervectors.

    Default dimension dim -- 1000
    """
    
    def __init__(self, name, d=1000):
        # self.dim = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION']
        self.item_memory_name = name
        self.dim = d
        self.item_count = 0
        self.names = []
        self.memory = np.zeros((1, self.dim))

    def append(self, name, hd_vec):
        """Add hypervector to a memory."""

        if not self.item_count:
            self.memory[self.item_count, :] = hd_vec
        else:
            self.memory = np.append(self.memory, hd_vec, axis=0)

        self.names.append(name)

        self.item_count += 1

    def append_batch(self, names, d=1000):
        """Add batch of hypervectors."""

        set_dimension(vsa_dimension=d)

        for name in names:
            self.append(name, hd_vec=generate())

    def search(self, hd_vec_query, distance=True):
        """Return distances from query hypervector to every hypervector in the item memory."""
        
        result = np.zeros((1, self.item_count))

    

        for i in range(self.item_count):
            if distance:
                result[0, i] = 1 - sim(self.memory[i, :], hd_vec_query)
            else:
                result[0, i] = sim(self.memory[i, :], hd_vec_query)
        
        return result

    def get_names(self):
        """Get names of stored entities."""
        return self.names

    def get_im_name(self):
        """Get name of ItemMemory."""
        return self.item_memory_name

    def get_name(self, index):
        """Get name of entity by its index."""
        
        return self.names[index]

    def get_vector(self, name):
        """Get stored hypervector by name of entity it represents."""
        
        idx = self.names.index(name)
        return self.memory[idx, :].reshape(1, -1)
    
    def get_dim(self):
        """Get dim."""
        
        return self.dim
