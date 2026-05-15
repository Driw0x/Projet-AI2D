import utils
import ast_error_detection as aed
from differences_tag import Differences_tag
from evolution_code import Evolution_code

# Obtenir les différents tags nécessaires entre 2 codes
def comparaison(code1, code2):
    modif_base = aed.get_typology_based_code_error(code1, [code2])[1]
    modif_detaillee = utils.primary_code_error_two_prog(code1, code2)[1]
    
    tag = Differences_tag()

    for differences in modif_base:
        if differences.startswith("EXP_MISSING_ASSIGNMENT_MISSING"):
            tag.ajout_variable = True

        elif differences.startswith("LO_FOR_MISSING"):
            tag.ajout_boucle_for = True
        elif differences.startswith("LO_FOR_NUMBER_ITERATION_ERROR"):
            tag.modif_boucle_for_iteration = True

        elif differences.startswith("LO_WHILE_MISSING"):
            tag.ajout_boucle_while = True
        elif differences.startswith("LO_WHILE_NUMBER_ITERATION"):
            tag.modif_boucle_while_iteration = True

    for i in range(len(modif_detaillee)):
        if modif_detaillee[i][2].startswith("Module > For"):
            tag.modif_corps_boucle_for = True
        if modif_detaillee[i][2].startswith("Module > While"):
            tag.modif_corps_boucle_while = True
        if modif_detaillee[i][0].startswith("MISSING_IF_STATEMENT"):
            tag.ajout_if = True
        if modif_detaillee[i][2].startswith("Module > Call"):
            tag.modif_hors_struct = True

    return tag


# Obtenir toutes les modifs à analyser sur une série de code
def evo_code(codes):
    # Ajout du code vide au début
    codes = [""] + codes

    # Fonction pour mettre à jour les range de modifications
    def update_range(bool, liste_range, i):
        if bool:
            if len(liste_range) == 0:
                liste_range.append([i])
            elif len(liste_range[-1]) == 2:
                liste_range.append([i])
        else:
            if len(liste_range) != 0 and len(liste_range[-1]) == 1:
                liste_range[-1].append(i)

    if len(codes) <= 1:
        return
    
    # La structure à retourner
    evo = Evolution_code()
    
    # On compare 2 codes successifs
    for i in range(len(codes) - 1):
        code1 = codes[i]
        code2 = codes[i+1]

        tag = comparaison(code1, code2)

        # On gère les apparitions des structures de contrôle
        if evo.premiere_boucle_for == -1 and tag.ajout_boucle_for:
            evo.premiere_boucle_for = i+1
        if evo.premiere_boucle_while == -1 and tag.ajout_boucle_while:
            evo.premiere_boucle_while = i+1
        if evo.premiere_struct_if == -1 and tag.ajout_if:
            evo.premiere_struct_if = i+1

        # On met à jour les ranges
        update_range(tag.ajout_boucle_for or tag.modif_boucle_for_iteration or tag.modif_corps_boucle_for, evo.range_modif_for, i)
        update_range(tag.ajout_boucle_while or tag.modif_boucle_while_iteration or tag.modif_corps_boucle_while, evo.range_modif_while, i)
        update_range(tag.ajout_if or tag.modif_cond_if or tag.modif_corps_if, evo.range_modif_if, i)
        update_range(tag.modif_hors_struct, evo.range_modif_hors_struct, i)

    
    # Fin des modif sur les structures (ou hors struct) s'il y en avait des incomplètes
    if len(evo.range_modif_for) != 0 and len(evo.range_modif_for[-1]) == 1:
        evo.range_modif_for[-1].append(len(codes)-1)
    if len(evo.range_modif_while) != 0 and len(evo.range_modif_while[-1]) == 1:
        evo.range_modif_while[-1].append(len(codes)-1)
    if len(evo.range_modif_if) != 0 and len(evo.range_modif_if[-1]) == 1:
        evo.range_modif_if[-1].append(len(codes)-1)
    if len(evo.range_modif_hors_struct) != 0 and len(evo.range_modif_hors_struct[-1]) == 1:
        evo.range_modif_hors_struct[-1].append(len(codes)-1)

    return evo


if __name__ == "__main__":
    test = ["couleur(255,0,0)",\
            "couleur(255,0,0)\navancer(2)",\
            "couleur(255,0,0)\navancer(2)\nfor i in range(4):\n    tourner(1, 90)",\
            "couleur(255,0,0)\navancer(2)\nfor i in range(4):\n    tourner(1, 90)\nfor i in range(2):\n    avancer(2)",\
            "couleur(255,0,0)\navancer(2)\nfor i in range(4):\n    tourner(1, 90)\nfor i in range(2):\n    avancer(2)\navancer(4)",\
            "couleur(255,0,0)\navancer(2)\nfor i in range(4):\n    tourner(1, 90)\nfor i in range(4):\n    avancer(2)\ntourner(1,90)\navancer(3)"]
    print(evo_code(test))