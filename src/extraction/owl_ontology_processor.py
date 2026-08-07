from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


PART_ID = "part_1_ontology"

OWL_NAMESPACE = "http://www.w3.org/2002/07/owl#"
RDF_NAMESPACE = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
)
RDFS_NAMESPACE = (
    "http://www.w3.org/2000/01/rdf-schema#"
)

RDF_ABOUT = f"{{{RDF_NAMESPACE}}}about"
RDF_RESOURCE = f"{{{RDF_NAMESPACE}}}resource"
RDF_TYPE = f"{{{RDF_NAMESPACE}}}type"
RDF_DATATYPE = f"{{{RDF_NAMESPACE}}}datatype"

OWL_XML_ROOT = f"{{{OWL_NAMESPACE}}}Ontology"
RDF_XML_ROOT = f"{{{RDF_NAMESPACE}}}RDF"


@dataclass
class ExtractedOntologyItem:
    unit_type: str
    label: str
    content: str
    structured_data: dict[str, Any]
    confidence: float = 0.98


def _local_name(tag: str) -> str:
    """
    Return the local part of an XML element or attribute name.
    """
    if "}" in tag:
        return tag.rsplit("}", 1)[1]

    if ":" in tag:
        return tag.rsplit(":", 1)[1]

    return tag


def _compact_iri(
    iri: str | None,
) -> str | None:
    """
    Convert a full IRI or fragment into a readable entity label.
    """
    if not iri:
        return None

    if iri == "topObjectProperty":
        return "owl:topObjectProperty"

    if "#" in iri:
        fragment = iri.rsplit("#", 1)[1]

        if fragment:
            return fragment

    stripped = iri.rstrip("/")

    if "/" in stripped:
        fragment = stripped.rsplit("/", 1)[1]

        if fragment:
            return fragment

    return iri


def _element_iri(
    element: ET.Element | None,
) -> str | None:
    if element is None:
        return None

    for attribute_name in (
        "IRI",
        "abbreviatedIRI",
        RDF_ABOUT,
        RDF_RESOURCE,
    ):
        value = element.attrib.get(
            attribute_name
        )

        if value:
            return value

    return None


def _entity_label(
    element: ET.Element | None,
) -> str | None:
    return _compact_iri(
        _element_iri(element)
    )


def _xml_text(
    element: ET.Element,
) -> str:
    return ET.tostring(
        element,
        encoding="unicode",
    ).strip()


def _task_mapping() -> dict[str, Any]:
    return {
        "part_ids": [
            PART_ID,
        ],
        "source": "inferred",
        "confidence": 1.0,
        "evidence": [
            (
                "The unit was extracted from the "
                "submitted Part 1 ontology artifact."
            ),
        ],
    }


def _safe_integer(
    value: str | None,
) -> int | None:
    if value is None:
        return None

    try:
        return int(value.strip())
    except ValueError:
        return None


def _first_child(
    element: ET.Element,
    local_name: str,
) -> ET.Element | None:
    for child in element:
        if _local_name(child.tag) == local_name:
            return child

    return None


def _children_named(
    element: ET.Element,
    local_name: str,
) -> list[ET.Element]:
    return [
        child
        for child in element
        if _local_name(child.tag) == local_name
    ]


def _descendants_named(
    element: ET.Element,
    local_name: str,
) -> list[ET.Element]:
    return [
        descendant
        for descendant in element.iter()
        if _local_name(descendant.tag)
        == local_name
    ]


def _restriction_data_from_owl_xml(
    restriction: ET.Element,
) -> dict[str, Any]:
    restriction_type = _local_name(
        restriction.tag
    )

    property_element = None
    filler_element = None

    for child in restriction:
        child_name = _local_name(child.tag)

        if child_name in {
            "ObjectProperty",
            "DataProperty",
        } and property_element is None:
            property_element = child

        elif child_name in {
            "Class",
            "Datatype",
        } and filler_element is None:
            filler_element = child

    cardinality = _safe_integer(
        restriction.attrib.get("cardinality")
    )

    result: dict[str, Any] = {
        "axiom_type": (
            _owl_xml_restriction_type(
                restriction_type
            )
        ),
        "property": _entity_label(
            property_element
        ),
        "filler": _entity_label(
            filler_element
        ),
    }

    if cardinality is not None:
        result["cardinality"] = cardinality

    return result


def _owl_xml_restriction_type(
    element_name: str,
) -> str:
    mapping = {
        "ObjectExactCardinality": (
            "object_exact_cardinality"
        ),
        "ObjectMinCardinality": (
            "object_min_cardinality"
        ),
        "ObjectMaxCardinality": (
            "object_max_cardinality"
        ),
        "ObjectSomeValuesFrom": (
            "object_some_values_from"
        ),
        "ObjectAllValuesFrom": (
            "object_all_values_from"
        ),
        "DataExactCardinality": (
            "data_exact_cardinality"
        ),
        "DataMinCardinality": (
            "data_min_cardinality"
        ),
        "DataMaxCardinality": (
            "data_max_cardinality"
        ),
        "DataSomeValuesFrom": (
            "data_some_values_from"
        ),
        "DataAllValuesFrom": (
            "data_all_values_from"
        ),
    }

    return mapping.get(
        element_name,
        element_name.lower(),
    )


def _extract_owl_xml_declarations(
    root: ET.Element,
) -> list[ExtractedOntologyItem]:
    items: list[ExtractedOntologyItem] = []

    for declaration in _children_named(
        root,
        "Declaration",
    ):
        if not list(declaration):
            continue

        entity = list(declaration)[0]
        entity_type = _local_name(entity.tag)
        label = _entity_label(entity)

        if label is None:
            continue

        unit_type_map = {
            "Class": "ontology_class",
            "ObjectProperty": (
                "ontology_object_property"
            ),
            "DataProperty": (
                "ontology_datatype_property"
            ),
            "NamedIndividual": (
                "ontology_individual"
            ),
        }

        unit_type = unit_type_map.get(
            entity_type,
            "ontology_declaration",
        )

        items.append(
            ExtractedOntologyItem(
                unit_type=unit_type,
                label=label,
                content=_xml_text(
                    declaration
                ),
                structured_data={
                    "item_kind": "declaration",
                    "entity_type": (
                        entity_type
                    ),
                    "entity": label,
                    "iri": _element_iri(entity),
                    "serialisation": "owl_xml",
                },
            )
        )

    return items


def _extract_owl_xml_subclass_axioms(
    root: ET.Element,
) -> list[ExtractedOntologyItem]:
    items: list[ExtractedOntologyItem] = []

    restriction_names = {
        "ObjectExactCardinality",
        "ObjectMinCardinality",
        "ObjectMaxCardinality",
        "ObjectSomeValuesFrom",
        "ObjectAllValuesFrom",
        "DataExactCardinality",
        "DataMinCardinality",
        "DataMaxCardinality",
        "DataSomeValuesFrom",
        "DataAllValuesFrom",
    }

    for axiom in _children_named(
        root,
        "SubClassOf",
    ):
        children = list(axiom)

        if len(children) < 2:
            continue

        subject = _entity_label(
            children[0]
        )

        if subject is None:
            subject = (
                _local_name(children[0].tag)
            )

        expression = children[1]
        expression_name = _local_name(
            expression.tag
        )

        if expression_name == "Class":
            parent = _entity_label(
                expression
            )

            if parent is None:
                continue

            items.append(
                ExtractedOntologyItem(
                    unit_type=(
                        "ontology_subclass_axiom"
                    ),
                    label=(
                        f"{subject} subclass of "
                        f"{parent}"
                    ),
                    content=_xml_text(axiom),
                    structured_data={
                        "item_kind": "axiom",
                        "axiom_type": (
                            "subclass_of"
                        ),
                        "subject": subject,
                        "parent": parent,
                        "serialisation": (
                            "owl_xml"
                        ),
                    },
                )
            )

            continue

        if expression_name in restriction_names:
            restriction_data = (
                _restriction_data_from_owl_xml(
                    expression
                )
            )

            restriction_data.update(
                {
                    "item_kind": "axiom",
                    "subject": subject,
                    "serialisation": (
                        "owl_xml"
                    ),
                }
            )

            property_name = (
                restriction_data.get(
                    "property"
                )
            )

            label = (
                f"{subject}: "
                f"{restriction_data['axiom_type']}"
            )

            if property_name:
                label += f" on {property_name}"

            items.append(
                ExtractedOntologyItem(
                    unit_type=(
                        "ontology_restriction"
                    ),
                    label=label,
                    content=_xml_text(axiom),
                    structured_data=(
                        restriction_data
                    ),
                )
            )

            continue

        items.append(
            ExtractedOntologyItem(
                unit_type=(
                    "ontology_complex_subclass_axiom"
                ),
                label=(
                    f"{subject}: "
                    f"{expression_name}"
                ),
                content=_xml_text(axiom),
                structured_data={
                    "item_kind": "axiom",
                    "axiom_type": (
                        "complex_subclass_of"
                    ),
                    "subject": subject,
                    "expression_type": (
                        expression_name
                    ),
                    "serialisation": "owl_xml",
                },
                confidence=0.9,
            )
        )

    return items


def _extract_owl_xml_property_axioms(
    root: ET.Element,
) -> list[ExtractedOntologyItem]:
    items: list[ExtractedOntologyItem] = []

    configurations = {
        "SubObjectPropertyOf": {
            "unit_type": (
                "ontology_subproperty_axiom"
            ),
            "axiom_type": (
                "sub_object_property_of"
            ),
            "subject_key": "subproperty",
            "object_key": "superproperty",
        },
        "ObjectPropertyDomain": {
            "unit_type": (
                "ontology_property_domain"
            ),
            "axiom_type": (
                "object_property_domain"
            ),
            "subject_key": "property",
            "object_key": "domain",
        },
        "ObjectPropertyRange": {
            "unit_type": (
                "ontology_property_range"
            ),
            "axiom_type": (
                "object_property_range"
            ),
            "subject_key": "property",
            "object_key": "range",
        },
        "DataPropertyDomain": {
            "unit_type": (
                "ontology_property_domain"
            ),
            "axiom_type": (
                "data_property_domain"
            ),
            "subject_key": "property",
            "object_key": "domain",
        },
        "DataPropertyRange": {
            "unit_type": (
                "ontology_property_range"
            ),
            "axiom_type": (
                "data_property_range"
            ),
            "subject_key": "property",
            "object_key": "range",
        },
    }

    for element_name, configuration in (
        configurations.items()
    ):
        for axiom in _children_named(
            root,
            element_name,
        ):
            children = list(axiom)

            if len(children) < 2:
                continue

            subject = _entity_label(
                children[0]
            )
            target = _entity_label(
                children[1]
            )

            if subject is None or target is None:
                continue
            
            if (
                element_name == "SubObjectPropertyOf"
                and target == "owl:topObjectProperty"
            ):
                continue

            structured_data = {
                "item_kind": "axiom",
                "axiom_type": (
                    configuration[
                        "axiom_type"
                    ]
                ),
                configuration[
                    "subject_key"
                ]: subject,
                configuration[
                    "object_key"
                ]: target,
                "serialisation": "owl_xml",
            }

            items.append(
                ExtractedOntologyItem(
                    unit_type=configuration[
                        "unit_type"
                    ],
                    label=(
                        f"{subject}: "
                        f"{configuration['axiom_type']} "
                        f"{target}"
                    ),
                    content=_xml_text(axiom),
                    structured_data=(
                        structured_data
                    ),
                )
            )

    return items


def _extract_owl_xml_items(
    root: ET.Element,
) -> list[ExtractedOntologyItem]:
    return [
        *_extract_owl_xml_declarations(
            root
        ),
        *_extract_owl_xml_subclass_axioms(
            root
        ),
        *_extract_owl_xml_property_axioms(
            root
        ),
    ]


def _rdf_attribute_label(
    element: ET.Element | None,
    attribute_name: str,
) -> str | None:
    if element is None:
        return None

    return _compact_iri(
        element.attrib.get(attribute_name)
    )


def _rdf_restriction_data(
    restriction: ET.Element,
) -> dict[str, Any]:
    property_element = _first_child(
        restriction,
        "onProperty",
    )

    property_name = _rdf_attribute_label(
        property_element,
        RDF_RESOURCE,
    )

    some_values = _first_child(
        restriction,
        "someValuesFrom",
    )

    all_values = _first_child(
        restriction,
        "allValuesFrom",
    )

    on_class = _first_child(
        restriction,
        "onClass",
    )

    exact = _first_child(
        restriction,
        "qualifiedCardinality",
    )

    minimum = _first_child(
        restriction,
        "minQualifiedCardinality",
    )

    maximum = _first_child(
        restriction,
        "maxQualifiedCardinality",
    )

    unqualified_exact = _first_child(
        restriction,
        "cardinality",
    )

    unqualified_minimum = _first_child(
        restriction,
        "minCardinality",
    )

    unqualified_maximum = _first_child(
        restriction,
        "maxCardinality",
    )

    filler = None
    axiom_type = "restriction"
    cardinality = None

    if some_values is not None:
        axiom_type = "object_some_values_from"
        filler = _rdf_attribute_label(
            some_values,
            RDF_RESOURCE,
        )

    elif all_values is not None:
        axiom_type = "object_all_values_from"
        filler = _rdf_attribute_label(
            all_values,
            RDF_RESOURCE,
        )

    elif exact is not None:
        axiom_type = (
            "object_exact_cardinality"
        )
        cardinality = _safe_integer(
            exact.text
        )

    elif minimum is not None:
        axiom_type = (
            "object_min_cardinality"
        )
        cardinality = _safe_integer(
            minimum.text
        )

    elif maximum is not None:
        axiom_type = (
            "object_max_cardinality"
        )
        cardinality = _safe_integer(
            maximum.text
        )

    elif unqualified_exact is not None:
        axiom_type = "exact_cardinality"
        cardinality = _safe_integer(
            unqualified_exact.text
        )

    elif unqualified_minimum is not None:
        axiom_type = "min_cardinality"
        cardinality = _safe_integer(
            unqualified_minimum.text
        )

    elif unqualified_maximum is not None:
        axiom_type = "max_cardinality"
        cardinality = _safe_integer(
            unqualified_maximum.text
        )

    if filler is None:
        filler = _rdf_attribute_label(
            on_class,
            RDF_RESOURCE,
        )

    result: dict[str, Any] = {
        "axiom_type": axiom_type,
        "property": property_name,
        "filler": filler,
    }

    if cardinality is not None:
        result["cardinality"] = cardinality

    return result


def _extract_rdf_class_items(
    root: ET.Element,
) -> list[ExtractedOntologyItem]:
    items: list[ExtractedOntologyItem] = []

    class_elements = [
        element
        for element in root
        if _local_name(element.tag)
        == "Class"
        and element.tag.startswith(
            f"{{{OWL_NAMESPACE}}}"
        )
    ]

    for class_element in class_elements:
        class_name = _compact_iri(
            class_element.attrib.get(
                RDF_ABOUT
            )
        )

        if class_name is None:
            continue

        items.append(
            ExtractedOntologyItem(
                unit_type="ontology_class",
                label=class_name,
                content=_xml_text(
                    class_element
                ),
                structured_data={
                    "item_kind": "declaration",
                    "entity_type": "Class",
                    "entity": class_name,
                    "iri": (
                        class_element.attrib.get(
                            RDF_ABOUT
                        )
                    ),
                    "serialisation": "rdf_xml",
                },
            )
        )

        for subclass_element in (
            _children_named(
                class_element,
                "subClassOf",
            )
        ):
            direct_parent = _compact_iri(
                subclass_element.attrib.get(
                    RDF_RESOURCE
                )
            )

            if direct_parent is not None:
                items.append(
                    ExtractedOntologyItem(
                        unit_type=(
                            "ontology_subclass_axiom"
                        ),
                        label=(
                            f"{class_name} subclass "
                            f"of {direct_parent}"
                        ),
                        content=_xml_text(
                            subclass_element
                        ),
                        structured_data={
                            "item_kind": "axiom",
                            "axiom_type": (
                                "subclass_of"
                            ),
                            "subject": class_name,
                            "parent": direct_parent,
                            "serialisation": (
                                "rdf_xml"
                            ),
                        },
                    )
                )

            restrictions = (
                _descendants_named(
                    subclass_element,
                    "Restriction",
                )
            )

            for restriction in restrictions:
                restriction_data = (
                    _rdf_restriction_data(
                        restriction
                    )
                )

                restriction_data.update(
                    {
                        "item_kind": "axiom",
                        "subject": class_name,
                        "serialisation": (
                            "rdf_xml"
                        ),
                    }
                )

                property_name = (
                    restriction_data.get(
                        "property"
                    )
                )

                label = (
                    f"{class_name}: "
                    f"{restriction_data['axiom_type']}"
                )

                if property_name:
                    label += (
                        f" on {property_name}"
                    )

                items.append(
                    ExtractedOntologyItem(
                        unit_type=(
                            "ontology_restriction"
                        ),
                        label=label,
                        content=_xml_text(
                            restriction
                        ),
                        structured_data=(
                            restriction_data
                        ),
                    )
                )

            intersection = next(
                (
                    descendant
                    for descendant
                    in subclass_element.iter()
                    if _local_name(
                        descendant.tag
                    )
                    == "intersectionOf"
                ),
                None,
            )

            if intersection is not None:
                members: list[str] = []

                for descendant in (
                    intersection.iter()
                ):
                    member = _compact_iri(
                        descendant.attrib.get(
                            RDF_ABOUT
                        )
                        or descendant.attrib.get(
                            RDF_RESOURCE
                        )
                    )

                    if (
                        member is not None
                        and member
                        not in members
                    ):
                        members.append(member)

                items.append(
                    ExtractedOntologyItem(
                        unit_type=(
                            "ontology_intersection_axiom"
                        ),
                        label=(
                            f"{class_name}: "
                            "intersection"
                        ),
                        content=_xml_text(
                            intersection
                        ),
                        structured_data={
                            "item_kind": "axiom",
                            "axiom_type": (
                                "intersection_of"
                            ),
                            "subject": class_name,
                            "members": members,
                            "serialisation": (
                                "rdf_xml"
                            ),
                        },
                        confidence=0.95,
                    )
                )

    return items


def _property_characteristics(
    property_element: ET.Element,
) -> list[str]:
    characteristics: list[str] = []

    for type_element in _children_named(
        property_element,
        "type",
    ):
        resource = _compact_iri(
            type_element.attrib.get(
                RDF_RESOURCE
            )
        )

        if (
            resource
            and resource
            not in characteristics
        ):
            characteristics.append(resource)

    return characteristics


def _extract_rdf_property_items(
    root: ET.Element,
) -> list[ExtractedOntologyItem]:
    items: list[ExtractedOntologyItem] = []

    for property_element in root:
        local_name = _local_name(
            property_element.tag
        )

        if local_name not in {
            "ObjectProperty",
            "DatatypeProperty",
        }:
            continue

        if not property_element.tag.startswith(
            f"{{{OWL_NAMESPACE}}}"
        ):
            continue

        property_name = _compact_iri(
            property_element.attrib.get(
                RDF_ABOUT
            )
        )

        if property_name is None:
            continue

        unit_type = (
            "ontology_object_property"
            if local_name
            == "ObjectProperty"
            else "ontology_datatype_property"
        )

        domain_element = _first_child(
            property_element,
            "domain",
        )

        range_element = _first_child(
            property_element,
            "range",
        )

        subproperty_element = _first_child(
            property_element,
            "subPropertyOf",
        )

        domain = _rdf_attribute_label(
            domain_element,
            RDF_RESOURCE,
        )

        property_range = (
            _rdf_attribute_label(
                range_element,
                RDF_RESOURCE,
            )
        )

        superproperty = (
            _rdf_attribute_label(
                subproperty_element,
                RDF_RESOURCE,
            )
        )

        characteristics = (
            _property_characteristics(
                property_element
            )
        )

        items.append(
            ExtractedOntologyItem(
                unit_type=unit_type,
                label=property_name,
                content=_xml_text(
                    property_element
                ),
                structured_data={
                    "item_kind": "declaration",
                    "entity_type": local_name,
                    "entity": property_name,
                    "iri": (
                        property_element.attrib.get(
                            RDF_ABOUT
                        )
                    ),
                    "domain": domain,
                    "range": property_range,
                    "superproperty": (
                        superproperty
                    ),
                    "characteristics": (
                        characteristics
                    ),
                    "serialisation": "rdf_xml",
                },
            )
        )

        if domain is not None:
            items.append(
                ExtractedOntologyItem(
                    unit_type=(
                        "ontology_property_domain"
                    ),
                    label=(
                        f"{property_name}: "
                        f"domain {domain}"
                    ),
                    content=_xml_text(
                        domain_element
                    ),
                    structured_data={
                        "item_kind": "axiom",
                        "axiom_type": (
                            "property_domain"
                        ),
                        "property": property_name,
                        "domain": domain,
                        "serialisation": (
                            "rdf_xml"
                        ),
                    },
                )
            )

        if property_range is not None:
            items.append(
                ExtractedOntologyItem(
                    unit_type=(
                        "ontology_property_range"
                    ),
                    label=(
                        f"{property_name}: "
                        f"range {property_range}"
                    ),
                    content=_xml_text(
                        range_element
                    ),
                    structured_data={
                        "item_kind": "axiom",
                        "axiom_type": (
                            "property_range"
                        ),
                        "property": property_name,
                        "range": property_range,
                        "serialisation": (
                            "rdf_xml"
                        ),
                    },
                )
            )

        if superproperty is not None:
            items.append(
                ExtractedOntologyItem(
                    unit_type=(
                        "ontology_subproperty_axiom"
                    ),
                    label=(
                        f"{property_name} "
                        f"subproperty of "
                        f"{superproperty}"
                    ),
                    content=_xml_text(
                        subproperty_element
                    ),
                    structured_data={
                        "item_kind": "axiom",
                        "axiom_type": (
                            "subproperty_of"
                        ),
                        "subproperty": (
                            property_name
                        ),
                        "superproperty": (
                            superproperty
                        ),
                        "serialisation": (
                            "rdf_xml"
                        ),
                    },
                )
            )

        for characteristic in characteristics:
            items.append(
                ExtractedOntologyItem(
                    unit_type=(
                        "ontology_property_characteristic"
                    ),
                    label=(
                        f"{property_name}: "
                        f"{characteristic}"
                    ),
                    content=_xml_text(
                        property_element
                    ),
                    structured_data={
                        "item_kind": "axiom",
                        "axiom_type": (
                            "property_characteristic"
                        ),
                        "property": property_name,
                        "characteristic": (
                            characteristic
                        ),
                        "serialisation": (
                            "rdf_xml"
                        ),
                    },
                )
            )

    return items


def _extract_rdf_xml_items(
    root: ET.Element,
) -> list[ExtractedOntologyItem]:
    return [
        *_extract_rdf_class_items(root),
        *_extract_rdf_property_items(root),
    ]


def _deduplicate_items(
    items: list[ExtractedOntologyItem],
) -> list[ExtractedOntologyItem]:
    unique_items: list[
        ExtractedOntologyItem
    ] = []

    seen: set[
        tuple[str, str, str]
    ] = set()

    for item in items:
        key = (
            item.unit_type,
            item.label,
            item.content,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_items.append(item)

    return unique_items


def _build_unit(
    *,
    item: ExtractedOntologyItem,
    artifact_id: str,
    unit_index: int,
) -> dict[str, Any]:
    unit_id = (
        f"{artifact_id}_unit_{unit_index:03d}"
    )

    block_type = (
        "xml"
        if item.structured_data.get(
            "item_kind"
        )
        == "declaration"
        else "formal_statement"
    )

    return {
        "unit_id": unit_id,
        "unit_type": item.unit_type,
        "label": item.label,
        "task_mapping": _task_mapping(),
        "content_blocks": [
            {
                "block_id": (
                    f"{unit_id}_content"
                ),
                "block_type": block_type,
                "content": item.content,
                "language": "xml",
            }
        ],
        "structured_data": (
            item.structured_data
        ),
        "extraction_confidence": (
            item.confidence
        ),
    }


def _build_diagnostics(
    *,
    content: str,
    parse_error: str | None,
    serialisation: str | None,
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    diagnostic_index = 1

    if not content.strip():
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_"
                    f"{diagnostic_index:03d}"
                ),
                "severity": "error",
                "diagnostic_type": (
                    "empty_ontology_content"
                ),
                "message": (
                    "No textual ontology content "
                    "was available."
                ),
                "related_unit_ids": [],
            }
        )

        return diagnostics

    if parse_error is not None:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_"
                    f"{diagnostic_index:03d}"
                ),
                "severity": "error",
                "diagnostic_type": (
                    "ontology_xml_parse_error"
                ),
                "message": (
                    "The ontology XML could not be "
                    f"parsed: {parse_error}"
                ),
                "related_unit_ids": [],
            }
        )

        return diagnostics

    if serialisation is None:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_"
                    f"{diagnostic_index:03d}"
                ),
                "severity": "error",
                "diagnostic_type": (
                    "unsupported_ontology_"
                    "serialisation"
                ),
                "message": (
                    "The ontology root element was "
                    "not recognised as OWL/XML or "
                    "RDF/XML."
                ),
                "related_unit_ids": [],
            }
        )

        return diagnostics

    if not units:
        diagnostics.append(
            {
                "diagnostic_id": (
                    f"diagnostic_"
                    f"{diagnostic_index:03d}"
                ),
                "severity": "warning",
                "diagnostic_type": (
                    "no_ontology_units_extracted"
                ),
                "message": (
                    "The ontology XML was parsed, "
                    "but no supported ontology units "
                    "were extracted."
                ),
                "related_unit_ids": [],
            }
        )

    return diagnostics


def process_owl_ontology_artifact(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    artifact_id = artifact["artifact_id"]
    artifact_type = artifact["artifact_type"]
    content = artifact["content"][
        "raw_content"
    ]

    root: ET.Element | None = None
    parse_error: str | None = None
    serialisation: str | None = None

    if content.strip():
        try:
            root = ET.fromstring(content)
        except ET.ParseError as error:
            parse_error = str(error)

    extracted_items: list[
        ExtractedOntologyItem
    ] = []

    if root is not None:
        if root.tag == OWL_XML_ROOT:
            serialisation = "owl_xml"
            extracted_items = (
                _extract_owl_xml_items(root)
            )

        elif root.tag == RDF_XML_ROOT:
            serialisation = "rdf_xml"
            extracted_items = (
                _extract_rdf_xml_items(root)
            )

    extracted_items = _deduplicate_items(
        extracted_items
    )

    units = [
        _build_unit(
            item=item,
            artifact_id=artifact_id,
            unit_index=index,
        )
        for index, item in enumerate(
            extracted_items,
            start=1,
        )
    ]

    diagnostics = _build_diagnostics(
        content=content,
        parse_error=parse_error,
        serialisation=serialisation,
        units=units,
    )

    contains_error = any(
        diagnostic["severity"] == "error"
        for diagnostic in diagnostics
    )

    if contains_error and units:
        status = "partial"
    elif contains_error:
        status = "failed"
    elif units:
        status = "success"
    else:
        status = "partial"

    checks = [
        {
            "check_type": "owl_parse",
            "status": (
                "passed"
                if (
                    root is not None
                    and serialisation
                    is not None
                )
                else "failed"
            ),
            "tool": (
                "xml.etree.ElementTree"
            ),
            "summary": (
                (
                    "Parsed ontology as "
                    f"{serialisation}."
                )
                if serialisation
                else (
                    "The ontology XML was not "
                    "successfully recognised."
                )
            ),
        },
        {
            "check_type": (
                "owl_structural_extraction"
            ),
            "status": (
                "passed"
                if units
                else "failed"
            ),
            "tool": (
                "assessment-feedback-"
                "preprocessor"
            ),
            "summary": (
                f"Extracted {len(units)} "
                "ontology unit(s)."
            ),
        },
        {
            "check_type": (
                "owl_consistency"
            ),
            "status": "not_run",
            "tool": None,
            "summary": (
                "An OWL reasoner was not run, "
                "so logical consistency was not "
                "verified."
            ),
        },
    ]

    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "processing": {
            "status": status,
            "extraction_mode": "parser",
            "detected_content_type": (
                serialisation
            ),
            "tools": [
                {
                    "name": (
                        "xml.etree.ElementTree"
                    ),
                    "version": None,
                    "purpose": (
                        "Parsing OWL/XML and RDF/XML "
                        "ontology documents."
                    ),
                },
                {
                    "name": (
                        "assessment-feedback-"
                        "preprocessor"
                    ),
                    "version": "0.3.0",
                    "purpose": (
                        "Normalisation of ontology "
                        "declarations, hierarchy "
                        "axioms, restrictions, "
                        "domains, ranges, and "
                        "property characteristics."
                    ),
                },
            ],
            "checks": checks,
            "diagnostics": diagnostics,
        },
        "units": units,
    }