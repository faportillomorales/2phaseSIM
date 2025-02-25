import numpy as np

def metodo_secante(f0, f1, x0, x1, it , tol=1e-10):
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
    
    # if abs(f1 - f0) < 1e-12:
    #     print("Erro: divisão por zero na secante!")
    #     return None
    
    x2 = x1 - f1 * (x1 - x0) / (f0 - f0)
    
    if f1 < tol:
        print('secante converged')
        return x2  # Convergiu para a raiz
    
    return x2