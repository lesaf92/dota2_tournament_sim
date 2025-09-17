import math

class Glicko2Rating:
    """Stores the rating, deviation, and volatility for a single player/team."""
    def __init__(self, mu=1500.0, phi=350.0, sigma=0.06):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    def __repr__(self):
        return f"Glicko2Rating(mu={self.mu:.2f}, phi={self.phi:.2f}, sigma={self.sigma:.3f})"

class Glicko2:
    """
    The main class for calculating Glicko-2 ratings.
    """
    def __init__(self, mu=1500.0, phi=350.0, sigma=0.06, tau=0.5, epsilon=0.000001):
        self._default_mu = mu
        self._default_phi = phi
        self._default_sigma = sigma
        self.tau = tau
        self.epsilon = epsilon
        self._SCALE = 173.7178

    def create_rating(self, mu=None, phi=None, sigma=None):
        """Creates a new Glicko2Rating object with default values."""
        if mu is None: mu = self._default_mu
        if phi is None: phi = self._default_phi
        if sigma is None: sigma = self._default_sigma
        return Glicko2Rating(mu, phi, sigma)

    def rate_1vs1(self, rating1, rating2, drawn=False):
        """Calculates the new ratings after a single 1v1 match."""
        if drawn:
            score1, score2 = 0.5, 0.5
        else:
            score1, score2 = 1.0, 0.0

        # Step 2: Convert ratings to Glicko-2 scale
        mu1, phi1 = self._to_glicko2_scale(rating1.mu, rating1.phi)
        mu2, phi2 = self._to_glicko2_scale(rating2.mu, rating2.phi)

        # Step 3, 4, 5: Calculate v, delta, and update sigma
        v1 = self._calculate_v(mu1, phi2, rating1.sigma)
        delta1 = self._calculate_delta(v1, mu1, phi2, score1)
        new_sigma1 = self._update_volatility(delta1, phi1, v1, rating1.sigma)

        v2 = self._calculate_v(mu2, phi1, rating2.sigma)
        delta2 = self._calculate_delta(v2, mu2, phi1, score2)
        new_sigma2 = self._update_volatility(delta2, phi2, v2, rating2.sigma)

        # Step 6: Update rating and RD
        new_phi1_g2 = self._update_phi(phi1, new_sigma1, v1)
        new_mu1_g2 = self._update_mu(mu1, new_phi1_g2, phi2, score1)

        new_phi2_g2 = self._update_phi(phi2, new_sigma2, v2)
        new_mu2_g2 = self._update_mu(mu2, new_phi2_g2, phi1, score2)

        # Step 8: Convert back to Glicko-1 scale
        new_mu1, new_phi1 = self._to_glicko1_scale(new_mu1_g2, new_phi1_g2)
        new_mu2, new_phi2 = self._to_glicko1_scale(new_mu2_g2, new_phi2_g2)

        return Glicko2Rating(new_mu1, new_phi1, new_sigma1), Glicko2Rating(new_mu2, new_phi2, new_sigma2)

    # --- Internal Calculation Methods ---
    def _to_glicko2_scale(self, mu, phi):
        return (mu - self._default_mu) / self._SCALE, phi / self._SCALE

    def _to_glicko1_scale(self, mu_g2, phi_g2):
        return self._default_mu + mu_g2 * self._SCALE, phi_g2 * self._SCALE

    def _g(self, phi):
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

    def _E(self, mu, mu_opponent, phi_opponent):
        return 1 / (1 + math.exp(-self._g(phi_opponent) * (mu - mu_opponent)))

    def _calculate_v(self, mu, phi_opponent, sigma):
        g_phi_opp = self._g(phi_opponent)
        E = self._E(mu, 0, phi_opponent) # Simplified mu_opponent=0 in glicko2 scale
        return 1 / (g_phi_opp**2 * E * (1 - E))

    def _calculate_delta(self, v, mu, phi_opponent, score):
        return v * self._g(phi_opponent) * (score - self._E(mu, 0, phi_opponent))

    def _update_volatility(self, delta, phi, v, sigma):
        a = math.log(sigma**2)

        def f(x):
            ex = math.exp(x)
            term1 = (ex * (delta**2 - phi**2 - v - ex)) / (2 * (phi**2 + v + ex)**2)
            term2 = (x - a) / self.tau**2
            return term1 - term2

        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fA, fB = f(A), f(B)
        while abs(B - A) > self.epsilon:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB < 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC

        return math.exp(A / 2)

    def _update_phi(self, phi, new_sigma, v):
        phi_star = math.sqrt(phi**2 + new_sigma**2)
        return 1 / math.sqrt(1 / phi_star**2 + 1 / v)

    def _update_mu(self, mu, new_phi, phi_opponent, score):
        return mu + new_phi**2 * self._g(phi_opponent) * (score - self._E(mu, 0, phi_opponent))
