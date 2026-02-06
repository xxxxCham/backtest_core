"""
Module-ID: ui.range_editor

Purpose: Interface Streamlit pour éditer les plages de paramètres visuellement.

Role in pipeline: configuration UI

Key components: render_range_editor, RangeEditorState

Inputs: config/indicator_ranges.toml via RangeManager

Outputs: Plages modifiées sauvegardées

Dependencies: streamlit, utils.range_manager

Conventions: Interface intuitive avec sliders et validation en temps réel.

Read-if: Modification de l'interface d'édition des plages.

Skip-if: Utilisation CLI ou programmatique.
"""

from pathlib import Path

import streamlit as st

from utils.range_manager import RangeManager


class RangeEditorState:
    """État de l'éditeur de plages."""

    @staticmethod
    def init():
        """Initialise l'état de session."""
        if "range_manager" not in st.session_state:
            st.session_state.range_manager = RangeManager()

        if "range_editor_category" not in st.session_state:
            st.session_state.range_editor_category = None

        if "range_editor_modified" not in st.session_state:
            st.session_state.range_editor_modified = False

        if "range_editor_search" not in st.session_state:
            st.session_state.range_editor_search = ""


def render_range_editor():
    """
    Affiche l'interface d'édition des plages de paramètres.

    Cette interface permet de:
    - Visualiser toutes les plages définies
    - Modifier les valeurs min/max/step/default
    - Sauvegarder les modifications
    - Créer des backups automatiques
    """
    RangeEditorState.init()

    st.title("⚙️ Éditeur de Plages de Paramètres")
    st.markdown("---")

    manager: RangeManager = st.session_state.range_manager

    # Header avec statistiques
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_categories = len(manager.get_all_categories())
        st.metric("Catégories", total_categories)

    with col2:
        total_params = sum(len(manager.get_category_params(cat))
                          for cat in manager.get_all_categories())
        st.metric("Paramètres", total_params)

    with col3:
        if st.session_state.range_editor_modified:
            st.metric("Statut", "Modifié", delta="Non sauvegardé")
        else:
            st.metric("Statut", "Synchronisé")

    with col4:
        config_path = Path(manager.config_path).name
        st.metric("Fichier", config_path)

    st.markdown("---")

    # Barre de recherche
    search_term = st.text_input(
        "🔍 Rechercher un paramètre",
        value=st.session_state.range_editor_search,
        placeholder="Ex: ema, period, rsi..."
    )
    st.session_state.range_editor_search = search_term

    # Sidebar pour sélection de catégorie
    with st.sidebar:
        st.header("📚 Catégories")

        categories = manager.get_all_categories()

        # Filtrer par recherche
        if search_term:
            filtered_categories = [
                cat for cat in categories
                if search_term.lower() in cat.lower() or
                any(search_term.lower() in param.lower()
                    for param in manager.get_category_params(cat))
            ]
        else:
            filtered_categories = categories

        # Sélecteur de catégorie
        if filtered_categories:
            selected_category = st.selectbox(
                "Sélectionner une catégorie",
                options=filtered_categories,
                index=0 if st.session_state.range_editor_category not in filtered_categories
                      else filtered_categories.index(st.session_state.range_editor_category),
                key="category_selector"
            )
            st.session_state.range_editor_category = selected_category
        else:
            st.warning("Aucune catégorie trouvée.")
            selected_category = None

        st.markdown("---")

        # Boutons d'action
        col_save, col_reload = st.columns(2)

        with col_save:
            if st.button(
                "💾 Sauvegarder",
                width="stretch",
                disabled=not st.session_state.range_editor_modified,
            ):
                try:
                    manager.save_ranges(backup=True)
                    st.session_state.range_editor_modified = False
                    st.success("✅ Plages sauvegardées !")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")

        with col_reload:
            if st.button("🔄 Recharger", width="stretch"):
                manager._load_ranges()
                st.session_state.range_editor_modified = False
                st.success("✅ Plages rechargées !")
                st.rerun()

        if st.button("📥 Exporter JSON", width="stretch"):
            export_path = Path("config/indicator_ranges_export.json")
            data = manager.export_to_dict()
            import json
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            st.success(f"✅ Exporté vers: {export_path}")

    # Contenu principal - Édition des paramètres
    if selected_category:
        st.header(f"📊 Catégorie: {selected_category}")

        params = manager.get_category_params(selected_category)

        if not params:
            st.info(f"Aucun paramètre dans la catégorie '{selected_category}'.")
            return

        # Filtrer les paramètres par recherche
        if search_term:
            params = [p for p in params if search_term.lower() in p.lower()]

        if not params:
            st.warning(f"Aucun paramètre trouvé avec '{search_term}'.")
            return

        st.write(f"**{len(params)} paramètre(s) trouvé(s)**")
        st.markdown("---")

        # Éditer chaque paramètre
        for param in params:
            range_cfg = manager.get_range(selected_category, param)

            if range_cfg is None:
                continue

            with st.expander(f"📌 {param}", expanded=True):
                st.markdown(f"*{range_cfg.description}*")

                # Afficher les valeurs actuelles
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Valeurs actuelles:**")
                    st.code(f"""
Min:     {range_cfg.min}
Max:     {range_cfg.max}
Step:    {range_cfg.step}
Default: {range_cfg.default}
                    """)

                with col2:
                    st.markdown("**Modifier:**")

                    # Options ou valeurs numériques
                    if range_cfg.options:
                        st.info("Type: Options prédéfinies")
                        st.write("Options:", ", ".join(range_cfg.options))

                        new_default = st.selectbox(
                            "Valeur par défaut",
                            options=range_cfg.options,
                            index=range_cfg.options.index(range_cfg.default)
                                  if range_cfg.default in range_cfg.options else 0,
                            key=f"default_{selected_category}_{param}"
                        )

                        if new_default != range_cfg.default:
                            if st.button("✅ Appliquer", key=f"apply_{selected_category}_{param}"):
                                manager.update_range(selected_category, param, default=new_default)
                                st.session_state.range_editor_modified = True
                                st.success("Modifié !")
                                st.rerun()

                    else:
                        # Valeurs numériques
                        new_min = st.number_input(
                            "Minimum",
                            value=float(range_cfg.min),
                            key=f"min_{selected_category}_{param}",
                            format="%.4f"
                        )

                        new_max = st.number_input(
                            "Maximum",
                            value=float(range_cfg.max),
                            key=f"max_{selected_category}_{param}",
                            format="%.4f"
                        )

                        new_step = st.number_input(
                            "Pas",
                            value=float(range_cfg.step),
                            min_value=0.0001,
                            key=f"step_{selected_category}_{param}",
                            format="%.4f"
                        )

                        new_default = st.number_input(
                            "Valeur par défaut",
                            value=float(range_cfg.default),
                            min_value=new_min,
                            max_value=new_max,
                            key=f"default_{selected_category}_{param}",
                            format="%.4f"
                        )

                        # Validation
                        valid = True
                        if new_min >= new_max:
                            st.error("❌ Min doit être < Max")
                            valid = False
                        if new_default < new_min or new_default > new_max:
                            st.error("❌ Default doit être entre Min et Max")
                            valid = False
                        if new_step <= 0:
                            st.error("❌ Step doit être > 0")
                            valid = False

                        # Bouton d'application
                        if valid:
                            changed = (
                                new_min != range_cfg.min or
                                new_max != range_cfg.max or
                                new_step != range_cfg.step or
                                new_default != range_cfg.default
                            )

                            if changed:
                                if st.button("✅ Appliquer les modifications",
                                           key=f"apply_{selected_category}_{param}"):
                                    manager.update_range(
                                        selected_category, param,
                                        min_val=new_min,
                                        max_val=new_max,
                                        step=new_step,
                                        default=new_default
                                    )
                                    st.session_state.range_editor_modified = True
                                    st.success("✅ Modifications appliquées !")
                                    st.rerun()
                            else:
                                st.info("Aucune modification détectée.")

                st.markdown("---")

    else:
        st.info("👈 Sélectionnez une catégorie dans la sidebar pour commencer.")


def render_range_editor_compact():
    """
    Version compacte de l'éditeur pour intégration dans d'autres pages.

    Affiche uniquement les contrôles essentiels sans header ni sidebar.
    """
    RangeEditorState.init()

    manager: RangeManager = st.session_state.range_manager

    st.subheader("⚙️ Édition Rapide des Plages")

    # Sélection catégorie
    categories = manager.get_all_categories()
    selected_category = st.selectbox(
        "Catégorie",
        options=categories,
        key="compact_category"
    )

    if selected_category:
        params = manager.get_category_params(selected_category)

        selected_param = st.selectbox(
            "Paramètre",
            options=params,
            key="compact_param"
        )

        if selected_param:
            range_cfg = manager.get_range(selected_category, selected_param)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                new_min = st.number_input("Min", value=float(range_cfg.min), key="compact_min")
            with col2:
                new_max = st.number_input("Max", value=float(range_cfg.max), key="compact_max")
            with col3:
                new_step = st.number_input("Step", value=float(range_cfg.step), key="compact_step")
            with col4:
                new_default = st.number_input("Default", value=float(range_cfg.default), key="compact_default")

            if st.button("✅ Appliquer", key="compact_apply"):
                manager.update_range(
                    selected_category, selected_param,
                    min_val=new_min, max_val=new_max,
                    step=new_step, default=new_default
                )
                manager.save_ranges(backup=True)
                st.success("✅ Modifications appliquées et sauvegardées !")


# Point d'entrée pour test standalone
if __name__ == "__main__":
    st.set_page_config(
        page_title="Éditeur de Plages",
        page_icon="⚙️",
        layout="wide"
    )
    render_range_editor()
