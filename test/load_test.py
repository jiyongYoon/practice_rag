from load_data import pdf_loader

pdf_name = 'ls_elec_safety.pdf'

pdf = pdf_loader.load_pdf(pdf_name) ## Document 객체

print(pdf[0])