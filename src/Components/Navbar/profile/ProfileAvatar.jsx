import React from 'react';
import './ProfileAvatar.css';

const ProfileAvatar = ({ username = 'User' }) => {
  const getInitials = (name) => {
    const initials = name.split(' ').map((n) => n[0]).join('');
    return initials.toUpperCase();
  };

  return (
    <div className="profile-avatar">
      {getInitials(username)}
    </div>
  );
};

export default ProfileAvatar;
