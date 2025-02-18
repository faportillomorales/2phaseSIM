import numpy as np

def metodo_secante(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Método da secante para encontrar a raiz de f(x) = 0.

    Parâmetros:
    f - Função f(x) a ser resolvida
    x0, x1 - Estimativas iniciais da raiz
    tol - Tolerância para convergência
    max_iter - Número máximo de iterações

    Retorna:
    x_aprox - Aproximação da raiz
    """
    for _ in range(max_iter):
        if abs(f(x1) - f(x0)) < 1e-12:
            print("Erro: divisão por zero na secante!")
            return None

        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        if abs(x2 - x1) < tol:
            print('secante converged')
            return x2  # Convergiu para a raiz

        x0, x1 = x1, x2

    print("Número máximo de iterações atingido.")
    return None