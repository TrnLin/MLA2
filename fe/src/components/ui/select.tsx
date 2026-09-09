// Adapted from shadcn/ui's Radix Select. Styling uses this app's theme tokens.
// Source: https://ui.shadcn.com/docs/components/radix/select (MIT).
import type { ComponentProps } from 'react';
import * as SelectPrimitive from '@radix-ui/react-select';
import './select.css';

const Select = SelectPrimitive.Root;
function SelectValue(props: ComponentProps<typeof SelectPrimitive.Value>) {
  return <SelectPrimitive.Value data-slot="select-value" {...props} />;
}

function Chevron({ up = false }: { up?: boolean }) {
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={up ? 'm6 15 6-6 6 6' : 'm6 9 6 6 6-6'} /></svg>;
}

function SelectTrigger({ className = '', children, ...props }: ComponentProps<typeof SelectPrimitive.Trigger>) {
  return <SelectPrimitive.Trigger data-slot="select-trigger" className={`select-trigger ${className}`} {...props}>
    {children}
    <SelectPrimitive.Icon asChild><Chevron /></SelectPrimitive.Icon>
  </SelectPrimitive.Trigger>;
}

function SelectContent({ className = '', children, position = 'popper', align = 'start', ...props }: ComponentProps<typeof SelectPrimitive.Content>) {
  return <SelectPrimitive.Portal>
    <SelectPrimitive.Content data-slot="select-content" className={`select-content ${className}`} position={position} align={align} sideOffset={8} collisionPadding={12} {...props}>
      <SelectPrimitive.ScrollUpButton className="select-scroll-button"><Chevron up /></SelectPrimitive.ScrollUpButton>
      <SelectPrimitive.Viewport className="select-viewport">{children}</SelectPrimitive.Viewport>
      <SelectPrimitive.ScrollDownButton className="select-scroll-button"><Chevron /></SelectPrimitive.ScrollDownButton>
    </SelectPrimitive.Content>
  </SelectPrimitive.Portal>;
}

function SelectItem({ className = '', children, ...props }: ComponentProps<typeof SelectPrimitive.Item>) {
  return <SelectPrimitive.Item data-slot="select-item" className={`select-item ${className}`} {...props}>
    <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
    <span className="select-item-indicator"><SelectPrimitive.ItemIndicator>
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg>
    </SelectPrimitive.ItemIndicator></span>
  </SelectPrimitive.Item>;
}

export { Select, SelectContent, SelectItem, SelectTrigger, SelectValue };
