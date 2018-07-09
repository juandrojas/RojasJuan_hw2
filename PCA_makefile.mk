analisisPCA.pdf : graficasAdicionales.pdf PCA.pdf Pressure.pdf analisisPCA.tex
	pdflatex analisisPCA.tex

graficasAdicionales.pdf : RojasJuan_PCA.py
	python3 RojasJuan_PCA.py

Pressure.pdf : RojasJuan_PCA.py
	python3 RojasJuan_PCA.py

PCA.pdf : RojasJuan_PCA.py
	python3 RojasJuan_PCA.py
