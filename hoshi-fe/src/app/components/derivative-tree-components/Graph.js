import React from 'react';
import { Node } from './Node';
import { Edge } from './Edge';

export const Graph = ({
  data,
  setSelectedNode = () => {},
  selectedNode = null,
}) => {
  const renderNode = (node, x, y, z, level = 0) => {
    const position = [x, y, z];
    const elements = [
      <Node
        key={node.id}
        node={node}
        position={position}
        setSelectedNode={setSelectedNode}
        selectedNode={selectedNode}
      />,
    ];

    if (node.children && node.children.length > 0) {
      const angleStep = (2 * Math.PI) / node.children.length;
      const radius = 2;
      node.children.forEach((child, index) => {
        const childX = x + radius * Math.cos(index * angleStep);
        const childY = y - 2;
        const childZ = z + radius * Math.sin(index * angleStep);
        elements.push(
          <Edge
            key={`${node.id}-${child.id}`}
            start={position}
            end={[childX, childY, childZ]}
          />
        );
        elements.push(...renderNode(child, childX, childY, childZ, level + 1));
      });
    }
    return elements;
  };

  return <>{renderNode(data, 0, 0, 0)}</>;
};
