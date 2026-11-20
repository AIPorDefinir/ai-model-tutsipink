"""
medicion.py - Mide el tiempo, el CPU y la memoria de una etapa.

Uso dentro del mismo proceso:

    with Medidor() as m:
        hacer_algo()
    print(m.resultado)   # {"segundos": ..., "cpu_s": ..., "cpu_pct": ..., "mem_max_mb": ...}

Uso con un proceso aparte (por ejemplo un script de Spark, que además arranca una JVM hija):

    p = subprocess.Popen([...])
    with Medidor(pid=p.pid):
        p.wait()

Cada 0.1 s se toma una muestra del proceso y de todos sus hijos (en Spark, la JVM). La memoria es el
máximo de la suma de la memoria residente (RSS) de esos procesos. El CPU es el tiempo de CPU
(usuario + sistema) que consumieron durante la etapa; cpu_pct = cpu_s / segundos × 100, así que
100 equivale a un núcleo ocupado todo el tiempo y 200 a dos núcleos.
"""
import os
import resource
import threading
import time

import psutil


class Medidor:
    def __init__(self, pid=None, intervalo=0.1):
        self.pid = pid or os.getpid()
        self.externo = pid is not None
        self.intervalo = intervalo
        self.raiz = psutil.Process(self.pid)
        self.cpu = {}
        self.cpu_inicial = {}
        self.mem_max = 0
        self._parar = threading.Event()
        self.resultado = {}

    def _procesos(self):
        try:
            return [self.raiz] + self.raiz.children(recursive=True)
        except psutil.Error:
            return []

    def _muestra(self):
        rss = 0
        for p in self._procesos():
            try:
                with p.oneshot():
                    rss += p.memory_info().rss
                    t = p.cpu_times()
                    self.cpu[p.pid] = t.user + t.system
            except psutil.Error:
                pass
        self.mem_max = max(self.mem_max, rss)

    def _ciclo(self):
        while not self._parar.is_set():
            self._muestra()
            self._parar.wait(self.intervalo)

    def __enter__(self):
        self._uso_hijos = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._muestra()
        # En el propio proceso solo cuenta el CPU que se use a partir de aquí
        self.cpu_inicial = {} if self.externo else dict(self.cpu)
        self.t0 = time.perf_counter()
        self._hilo = threading.Thread(target=self._ciclo, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *exc):
        self._muestra()
        segundos = time.perf_counter() - self.t0
        self._parar.set()
        self._hilo.join()
        cpu_s = sum(v - self.cpu_inicial.get(pid, 0.0) for pid, v in self.cpu.items())
        if self.externo:
            # Los procesos que ya terminaron y se esperaron también cuentan en RUSAGE_CHILDREN
            u = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_hijos = (u.ru_utime - self._uso_hijos.ru_utime) + (u.ru_stime - self._uso_hijos.ru_stime)
            cpu_s = max(cpu_s, cpu_hijos)
        self.resultado = {"segundos": round(segundos, 4), "cpu_s": round(cpu_s, 4),
                          "cpu_pct": round(100 * cpu_s / segundos, 1) if segundos > 0 else 0.0,
                          "mem_max_mb": round(self.mem_max / 2**20, 1)}
        return False


def entorno():
    """Núcleos y memoria de la máquina donde se mide."""
    return {"nucleos_logicos": psutil.cpu_count(), "nucleos_fisicos": psutil.cpu_count(logical=False),
            "memoria_total_gb": round(psutil.virtual_memory().total / 1e9, 2)}
