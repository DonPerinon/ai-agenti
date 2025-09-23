# Simple JUT Agent

Krátky popis
------------
Simple JUT Agent je ľahký agent (bot) na vyhľadávanie a spracovanie informácií, postavený nad frameworkom *LangGraf* (LangGraph) a využívajúci vlastný search endpoint cez **SerpAPI MCP (custom SerpApi)** pre vyhľadávanie na webe.

Ciele projektu
--------------
- Rýchly prototyp agenta, ktorý dokáže spracovať dotazy, spustiť workflow v LangGraf a vrátiť relevantné výsledky z webu cez SerpAPI MCP.
- Modulárna architektúra: oddelené komponenty pre orchestráciu (LangGraf), vyhľadávanie (SerpAPI MCP) a spracovanie odpovede.

Hlavné funkcie
--------------
- Integrácia s LangGraf (graph/workflow framework)
- Vyhľadávanie pomocou SerpAPI MCP (custom SerpApi endpoint)
- Jednoduchá konfigurácia cez environmentálne premenné

Workflow
--------
- Ústredným prvkom riešenia je **supervisor node**, ktorý riadi tok dát a rozhoduje, ktoré kroky sa vykonajú.
- Supervisor podľa potreby využíva **tools nodes**, napríklad *researcher node* (napojený na SerpAPI), ktorý dohľadáva relevantné informácie a vracia ich späť na supervisor.
- Následne supervisor poverí **speaker node**, ktorý je zodpovedný za uhladenie a finálnu formuláciu odpovede.

Prehľad použitia
----------------
- V súbore `docker-compose.yml` je potrebné doplniť požadované environmentálne premenné.
- Riešenie sa spustí príkazom:

```bash
docker compose up -d
```

- Jednoduché webové UI je dostupné na adrese: [http://0.0.0.0:5000](http://0.0.0.0:5000)

Technológie
-----------
- LangGraf (framework pre graph/workflows)
- SerpAPI MCP (custom SerpApi endpoint pre vyhľadávanie)
- Python 3.12+

---