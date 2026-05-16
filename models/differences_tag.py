class Differences_tag:
    # Les tags que l'on va regarder pour les différences entre 2 codes successifss
    def __init__(self):
            self.ajout_variable = False

            self.ajout_boucle_for = False
            self.modif_corps_boucle_for = False
            self.modif_boucle_for_iteration = False

            self.ajout_if = False
            self.modif_cond_if = False
            self.modif_corps_if = False

            self.ajout_boucle_while = False
            self.modif_corps_boucle_while = False
            self.modif_boucle_while_iteration = False

            self.modif_hors_struct = False